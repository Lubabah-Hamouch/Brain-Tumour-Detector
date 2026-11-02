"""In-process background jobs for slow work (segmentation inference).

A single worker thread runs jobs one at a time in submission order, which is
also what the GPU wants: two segmentations in parallel would only compete for
memory. Job state lives in memory, so it is lost on restart. That is enough
for a single-user research demo; a multi-user deployment would use a real
queue (and persistent storage for results) instead.

Job lifecycle: queued -> running -> complete | failed. While running, the job
records the real stage reported by the work function, its own progress, and
an overall progress weighted by typical stage durations.
"""
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from backend.services.errors import ApiError


@dataclass
class Stage:
    name: str
    label: str
    weight: float  # share of total work, for overall progress
    status: str = "pending"  # pending | running | complete | failed
    fraction: float = 0.0
    detail: str = ""
    started: float | None = None
    finished: float | None = None


@dataclass
class Job:
    id: str
    kind: str
    stages: list
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: dict | None = None
    error: dict | None = None

    def snapshot(self, position=None):
        now = time.time()
        current = next((s for s in self.stages if s.status == "running"), None)
        done_weight = sum(s.weight for s in self.stages if s.status == "complete")
        if current:
            done_weight += current.weight * current.fraction
        total_weight = sum(s.weight for s in self.stages) or 1.0
        progress = 1.0 if self.status == "complete" else min(done_weight / total_weight, 0.999)
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "queue_position": position,
            "stage": current.name if current else None,
            "stage_label": current.label if current else None,
            "stage_detail": current.detail if current else None,
            "progress": round(progress, 4),
            "stages": [{"name": s.name, "label": s.label, "status": s.status,
                        "fraction": round(s.fraction, 4), "detail": s.detail,
                        "seconds": round((s.finished or now) - s.started, 2) if s.started else None}
                       for s in self.stages],
            "created_at": self.created,
            "started_at": self.started,
            "finished_at": self.finished,
            "elapsed_seconds": round((self.finished or now) - self.started, 2) if self.started else 0.0,
            "result": self.result,
            "error": self.error,
        }


class JobStore:
    def __init__(self, keep=32, workers=1):
        self.keep = keep
        self._jobs = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")

    def submit(self, kind, stages, work):
        """Queue work(progress) and return the job id. stages: [(name, label, weight)].

        work receives progress(stage_name, fraction, detail="") and returns the result dict.
        An ApiError raised by work is reported with its code; anything else as internal_error.
        """
        job = Job(id=uuid.uuid4().hex, kind=kind, stages=[Stage(n, l, w) for n, l, w in stages])
        with self._lock:
            self._jobs[job.id] = job
            self._evict()
        self._executor.submit(self._run, job, work)
        return job.id

    def add_complete(self, kind, stages, result):
        """Record a job whose result is already known (a pre-computed sample) as complete, without
        queueing any work. Its stages are complete and took no time."""
        now = time.time()
        job = Job(id=uuid.uuid4().hex, kind=kind, created=now, started=now, finished=now, status="complete",
                  result=result, stages=[Stage(n, l, w, status="complete", fraction=1.0, started=now, finished=now)
                                         for n, l, w in stages])
        with self._lock:
            self._jobs[job.id] = job
            self._evict()
        return job.id

    def get(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            queued = [j.id for j in self._jobs.values() if j.status == "queued"]
            position = queued.index(job_id) + 1 if job.status == "queued" else None
            return job.snapshot(position)

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _evict(self):
        # Drop the oldest finished jobs beyond `keep`; never drop queued or running work.
        finished = [k for k, j in self._jobs.items() if j.status in ("complete", "failed")]
        for key in finished[: max(0, len(self._jobs) - self.keep)]:
            del self._jobs[key]

    def _progress(self, job):
        def report(stage_name, fraction, detail=""):
            now = time.time()
            with self._lock:
                for s in job.stages:
                    if s.name == stage_name:
                        if s.status == "pending":
                            s.status, s.started = "running", now
                        s.fraction, s.detail = max(0.0, min(1.0, fraction)), detail
                    elif s.status == "running":  # a later stage started: close the previous one
                        s.status, s.fraction, s.finished = "complete", 1.0, now
        return report

    def _run(self, job, work):
        with self._lock:
            job.status, job.started = "running", time.time()
        try:
            result = work(self._progress(job))
            status, payload = "complete", {"result": result}
        except ApiError as exc:
            status, payload = "failed", {"error": {"code": exc.code, "message": exc.message}}
        except Exception as exc:  # report, never crash the worker
            traceback.print_exc()
            status, payload = "failed", {"error": {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}}
        now = time.time()
        with self._lock:
            for s in job.stages:
                if s.status == "running":
                    s.status, s.finished = ("complete" if status == "complete" else "failed"), now
                    if status == "complete":
                        s.fraction = 1.0
                elif s.status == "pending" and status == "complete":  # stage the work never reported
                    s.status, s.fraction, s.started, s.finished = "complete", 1.0, now, now
            job.status, job.finished = status, now
            job.result, job.error = payload.get("result"), payload.get("error")
