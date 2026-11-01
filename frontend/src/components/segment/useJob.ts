"use client";

import { useEffect, useState } from "react";

import { getJob, type JobStatus } from "@/lib/api";

export const POLL_INTERVAL_MS = 500;
const MAX_FAILED_POLLS = 5;

export type JobPoll = {
  /** The latest snapshot, and when (performance.now()) it arrived. */
  job: JobStatus | null;
  receivedAt: number;
  /** Set when the API stopped answering; the job itself may still be running. */
  lostContact: string | null;
};

/**
 * Polls a background job until it completes or fails. Each request waits for the previous one,
 * so a slow server is never flooded; a few failed polls in a row are tolerated before giving up.
 * Mount with `key={statusUrl}` so a new job starts from a clean state.
 */
export function useJob(statusUrl: string | null, intervalMs = POLL_INTERVAL_MS): JobPoll {
  const [poll, setPoll] = useState<JobPoll>({ job: null, receivedAt: 0, lostContact: null });

  useEffect(() => {
    if (!statusUrl) return;
    const controller = new AbortController();
    let timer = 0;
    let failures = 0;

    const tick = async () => {
      try {
        const job = await getJob(statusUrl, controller.signal);
        failures = 0;
        setPoll({ job, receivedAt: performance.now(), lostContact: null });
        if (job.status === "complete" || job.status === "failed") return;
      } catch (error) {
        if (controller.signal.aborted) return;
        failures += 1;
        if (failures >= MAX_FAILED_POLLS) {
          setPoll((p) => ({ ...p, lostContact: (error as Error).message }));
          return;
        }
      }
      timer = window.setTimeout(tick, intervalMs);
    };
    tick();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [statusUrl, intervalMs]);

  return poll;
}

/** performance.now(), updated every animation frame while `running`. */
export function useNow(running: boolean) {
  const [now, setNow] = useState(() => (typeof performance === "undefined" ? 0 : performance.now()));
  useEffect(() => {
    if (!running) return;
    let frame = requestAnimationFrame(function step(t) {
      setNow(t);
      frame = requestAnimationFrame(step);
    });
    return () => cancelAnimationFrame(frame);
  }, [running]);
  return now;
}
