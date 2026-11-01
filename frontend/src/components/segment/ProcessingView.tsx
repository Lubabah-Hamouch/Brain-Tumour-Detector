"use client";

import Image from "next/image";
import { useEffect } from "react";

import { Button } from "@/components/Button";
import { CheckIcon, ErrorIcon } from "@/components/icons";
import ws from "@/components/workspace/Workspace.module.css";
import type { JobStage, SegmentationResult } from "@/lib/api";
import { formatShape } from "@/lib/studyFiles";

import styles from "./Segment.module.css";
import type { Run } from "./types";
import { POLL_INTERVAL_MS, useJob, useNow } from "./useJob";

type Props = {
  run: Run;
  onComplete: (result: SegmentationResult) => void;
  onShowResults: () => void;
  onBack: () => void;
};

type Row = { name: string; detail: string; state: "done" | "active" | "pending" | "failed"; seconds: number | null };

const STAGE_DETAIL: Record<string, string> = {
  preprocessing: "Z-score inside the brain, crop to the brain",
  inference: "5 fold models · 128³ sliding window",
  postprocessing: "ET clean-up, label map, viewer slices",
};

export function ProcessingView({ run, onComplete, onShowResults, onBack }: Props) {
  const { job, receivedAt, lostContact } = useJob(run.statusUrl);
  const finished = job?.status === "complete" || job?.status === "failed" || Boolean(lostContact);
  // The clock stops when the job finishes, so the elapsed time freezes at the finish.
  const now = useNow(!finished);

  const result = job?.status === "complete" ? job.result : null;
  useEffect(() => {
    if (result) onComplete(result);
  }, [result, onComplete]);

  const failed = job?.status === "failed" || Boolean(lostContact);
  const uploading = run.uploadSeconds === null;
  const elapsed = Math.max(0, (now - run.startedAt) / 1000);
  const sinceSnapshot = job ? Math.max(0, (now - receivedAt) / 1000) : 0;
  const shape = run.header ? formatShape(run.header) : "";

  // First row: the upload and the server's synchronous checks (timed here); then the job's own stages.
  const rows: Row[] = [
    {
      name: "Upload and check",
      detail: `4 sequences${shape ? ` · ${shape}` : ""}`,
      state: uploading ? "active" : "done",
      seconds: uploading ? elapsed : run.uploadSeconds,
    },
    ...(job?.stages ?? []).map((s: JobStage): Row => {
      const state = s.status === "complete" ? "done" : s.status === "running" ? "active" : s.status === "failed" ? "failed" : "pending";
      const live = state === "active" && s.seconds !== null && !finished ? s.seconds + sinceSnapshot : s.seconds;
      return {
        name: s.label,
        detail: state === "active" || state === "failed" ? s.detail || STAGE_DETAIL[s.name] : s.name === "inference" && state === "done" ? "5-fold ensemble · 128³ sliding window" : STAGE_DETAIL[s.name] ?? s.detail,
        state,
        seconds: state === "pending" ? null : live,
      };
    }),
  ];
  if (!job) {
    for (const [name, label] of [["preprocessing", "Preprocess"], ["inference", "Inference"], ["postprocessing", "Post-process and encode"]]) {
      rows.push({ name: label, detail: STAGE_DETAIL[name], state: "pending", seconds: null });
    }
  }

  const progress = job ? job.progress : 0;
  const active = rows.find((r) => r.state === "active");
  const title = failed
    ? "Segmentation failed"
    : result
      ? "Segmentation complete"
      : uploading
        ? "Uploading"
        : job?.status === "queued"
          ? `Queued · position ${job.queue_position ?? 1}`
          : "Segmenting";
  const note = lostContact
    ? `The API stopped answering (${lostContact}). The job may still finish on the server.`
    : job?.status === "failed"
      ? job.error?.message ?? "The job failed."
      : result
        ? `Whole tumour ${result.regions.WT.volume_ml.toFixed(2)} mL · tumour core ${result.regions.TC.volume_ml.toFixed(2)} mL · enhancing ${result.regions.ET.volume_ml.toFixed(2)} mL`
        : uploading
          ? "Sending the four volumes; the server reads and checks them before queueing the job"
          : "Running the five fold models and averaging their predictions";
  const axial = result?.slices.axial;

  return (
    <div className={ws.screen}>
      <main className={`${ws.main} ${styles.processingMain}`}>
        <div className={`label ${ws.viewerHead} ${styles.narrow}`}>
          <span>
            Axial · FLAIR · <span className="mono">z {axial ? axial.index : run.preview?.index ?? "–"}</span>
          </span>
          <span>{failed ? "Stopped" : result ? "Prediction ready" : "Processing"}</span>
        </div>
        <div
          className={`${ws.viewer} ${styles.narrow} ${styles.processingViewer}`}
          role="img"
          aria-label={result ? "Axial FLAIR slice with the predicted tumour regions in red" : "Axial FLAIR slice being processed"}
        >
          <div className={ws.viewerInner}>
            {run.preview && (
              <Image
                className={ws.layer}
                src={run.preview.url}
                alt=""
                fill
                unoptimized
                style={{ opacity: axial ? 0 : 0.55 }}
              />
            )}
            {axial &&
              (["flair", "edema", "necrotic", "enhancing"] as const).map((layer) => (
                <Image key={layer} className={ws.layer} src={axial.layers[layer]} alt="" fill unoptimized />
              ))}
          </div>
          {!finished && <div className={styles.sweep} aria-hidden="true" />}
        </div>
        <div className={`${styles.narrow} ${styles.progress}`}>
          <div className={ws.track} aria-hidden="true">
            <div className={styles.progressFill} style={{ width: `${progress * 100}%` }} />
          </div>
          <div className={`caption ${styles.progressText}`}>
            <span>
              {failed ? "Stopped" : result ? "Complete" : active ? `${active.name} · ${active.detail}` : "Starting"}
            </span>
            <span className="mono">{Math.round(progress * 100)}%</span>
          </div>
        </div>
      </main>

      <aside className={ws.aside} aria-label="Progress">
        <section className={styles.statusBlock} aria-live="polite">
          <p className="label">{title}</p>
          <span className={`mono ${styles.elapsed}`}>
            {elapsed.toFixed(2)}
            <span className={styles.unit}> s</span>
          </span>
          <span className={failed ? styles.failNote : styles.statusNote}>{note}</span>
        </section>

        <section>
          {rows.map((row) => (
            <div key={row.name} className={styles.stage} data-state={row.state}>
              <span className={styles.stageMark}>
                {row.state === "done" && <CheckIcon />}
                {row.state === "failed" && <ErrorIcon />}
                {(row.state === "active" || row.state === "pending") && <span className={styles.dot} />}
              </span>
              <span className={styles.stageText}>
                <span>{row.name}</span>
                <span className="caption">{row.detail}</span>
              </span>
              <span className={`mono caption ${styles.stageTime}`}>
                {row.seconds !== null ? `${row.seconds.toFixed(2)} s` : ""}
              </span>
            </div>
          ))}
          <div className={ws.rule} />
        </section>

        <section className={ws.stack12}>
          {result && <Button block onClick={onShowResults}>View results</Button>}
          {failed && (
            <Button block variant="ghost" onClick={onBack}>
              Back to the files
            </Button>
          )}
          {!finished && (
            <button type="button" className={ws.textButton} onClick={onBack}>
              <span className="u">Stop watching and go back</span>
            </button>
          )}
        </section>

        <p className={ws.footnote}>
          Stages and times are reported by the server&apos;s job as it runs, polled every{" "}
          <span className="mono">{POLL_INTERVAL_MS}</span> ms; the upload is timed in the browser. On a CPU the
          inference stage takes about <span className="mono">3</span> minutes.
        </p>
      </aside>
    </div>
  );
}
