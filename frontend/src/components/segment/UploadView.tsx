"use client";

import { useId, useRef, useState, type DragEvent } from "react";

import { Button } from "@/components/Button";
import { CheckIcon, ErrorIcon } from "@/components/icons";
import { Term } from "@/components/Term";
import ws from "@/components/workspace/Workspace.module.css";
import { formatBytes } from "@/lib/imageCheck";
import { evaluate, MAX_VOLUME_MB, SEQUENCES, sequenceOf, type Sequence, type StudyFile } from "@/lib/studyFiles";

import styles from "./Segment.module.css";

type Props = {
  files: StudyFile[];
  evaluation: ReturnType<typeof evaluate>;
  sampleCase: string | null;
  loadingSample: boolean;
  runError: string | null;
  onAdd: (files: File[]) => void;
  onAssign: (id: string, sequence: Sequence) => void;
  onRemove: (id: string) => void;
  onSample: () => void;
  onRun: () => void;
};

export function UploadView({ files, evaluation, sampleCase, loadingSample, runError, ...on }: Props) {
  const inputId = useId();
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const missing = evaluation.slots.filter((s) => !s.file).map((s) => s.name);

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    if (event.dataTransfer.files.length) on.onAdd([...event.dataTransfer.files]);
  };

  return (
    <div className={ws.screen}>
      <main className={`${ws.main} ${styles.uploadMain}`}>
        <div className={styles.column}>
          <div className={styles.intro}>
            <p className="label">Segmentation · upload</p>
            <h1 className={ws.title}>Outline a brain tumour in 3D</h1>
            <p className={styles.lede}>
              The model reads four MRI scans of one patient (<Term k="sequences">FLAIR, T1, T1ce and T2</Term>) and marks
              the tumour&apos;s <Term k="regions">three regions</Term>, with their volumes and a 3D view.
            </p>
          </div>

          <section className={`${ws.tryPanel} ${styles.wide}`} aria-labelledby="try-segment">
            <div className={ws.tryText}>
              <p className="label">Try it now</p>
              <h2 id="try-segment" className={ws.tryTitle}>
                Run it on a sample patient
              </h2>
              <p className="caption">
                A patient from the held-out <Term k="brats">BraTS</Term> test set. One click uploads the four scans and runs
                the model, which takes seconds on a GPU.
              </p>
            </div>
            <div className={ws.tryOptions}>
              <button
                type="button"
                className={ws.tryOption}
                data-primary="true"
                disabled={loadingSample}
                onClick={on.onSample}
              >
                <span>{loadingSample ? "Loading the sample patient…" : "Run the sample patient"}</span>
                <span className="caption mono">BraTS2021_00003</span>
              </button>
            </div>
          </section>

          <div className={styles.own}>
            <p className="label">Or add your own scans</p>
            <p className="caption">
              Four <Term k="nifti">NIfTI</Term> files, one per scan type. Each is matched to its type from the file name;
              change any match that is wrong.
            </p>
          </div>

          <label
            htmlFor={inputId}
            className={`${ws.drop} ${styles.drop}`}
            data-state={dragging ? "dragging" : "empty"}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <span className={styles.dropContent}>
              <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
                <path d="M14 19V4M8 10l6-6 6 6M4 23h20" fill="none" stroke="currentColor" strokeWidth="1.4" />
              </svg>
              <span className={ws.dropTitle}>Drop NIfTI files here</span>
              <span className={ws.dropHint}>
                .nii or .nii.gz · up to <span className="mono">{MAX_VOLUME_MB}</span> MB each · all four at once or one
                by one
              </span>
            </span>
            <input
              ref={input}
              id={inputId}
              type="file"
              multiple
              accept=".nii,.gz,application/gzip"
              className={ws.fileInput}
              onChange={(event) => {
                if (event.target.files?.length) on.onAdd([...event.target.files]);
                event.target.value = "";
              }}
            />
          </label>
          <div className={styles.browse}>
            <Button variant="ghost" onClick={() => input.current?.click()}>
              Browse files
            </Button>
          </div>

          <section aria-label="Files" className={styles.table}>
            <div className={`label ${styles.tableHead}`}>
              <span>File</span>
              <span className={styles.right}>Size</span>
              <span>Sequence</span>
              <span>Check</span>
              <span />
            </div>
            {files.map((f) => {
              const status = evaluation.status.get(f.id)!;
              const seq = sequenceOf(f);
              const nifti = f.check.kind !== "rejected";
              return (
                <div key={f.id} className={styles.row} data-ok={status.ok} data-pending={Boolean(status.pending)}>
                  <span className={`mono ${styles.fileName}`} data-nifti={nifti} title={f.file.name}>
                    {f.file.name}
                  </span>
                  <span className={`mono ${styles.size}`}>{formatBytes(f.file.size)}</span>
                  <span className={styles.sequence}>
                    {nifti && (
                      <select
                        aria-label={`Sequence for ${f.file.name}`}
                        value={seq ?? ""}
                        data-problem={!status.ok && !status.pending}
                        onChange={(event) => on.onAssign(f.id, event.target.value as Sequence)}
                      >
                        {!seq && (
                          <option value="" disabled>
                            Choose…
                          </option>
                        )}
                        {SEQUENCES.map((s) => (
                          <option key={s} value={s}>
                            {s}
                          </option>
                        ))}
                      </select>
                    )}
                    <span className="caption">{status.source}</span>
                  </span>
                  <span className={styles.status}>
                    {status.ok && <CheckIcon />}
                    {!status.ok && !status.pending && <ErrorIcon />}
                    <span>{status.message}</span>
                  </span>
                  <button
                    type="button"
                    className={ws.remove}
                    aria-label={`Remove ${f.file.name}`}
                    onClick={() => on.onRemove(f.id)}
                  >
                    ×
                  </button>
                </div>
              );
            })}
            {!files.length && <p className={styles.empty}>No files yet</p>}
            <div className={ws.rule} />
          </section>
        </div>
      </main>

      <aside className={ws.aside} aria-label="Study status">
        <section>
          <p className={`label ${ws.sectionLabel}`}>
            <Term k="sequences">Sequences</Term>
          </p>
          {evaluation.slots.map((slot) => (
            <div key={slot.name} className={styles.slot} data-ok={Boolean(slot.file)}>
              <span className={styles.slotName}>{slot.name}</span>
              <span className={`mono ${styles.slotFile}`}>{slot.file?.file.name ?? "Missing"}</span>
              {slot.file && (
                <svg width="14" height="14" viewBox="0 0 14 14" aria-label="ready">
                  <path d="M2.5 7.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="1.5" />
                </svg>
              )}
            </div>
          ))}
          <div className={ws.rule} />
        </section>

        <section className={ws.stack12}>
          <Button block disabled={!evaluation.ready} onClick={on.onRun}>
            Run segmentation
          </Button>
          {runError ? (
            <span className={ws.runError} role="alert">
              <ErrorIcon />
              <span>{runError}</span>
            </span>
          ) : (
            <span className={ws.note} aria-live="polite">
              {evaluation.ready
                ? `All four sequences are ready${sampleCase ? ` (sample study ${sampleCase})` : ""}. Inference takes about 6 s on a GPU.`
                : `Still needed: ${missing.join(", ")}.`}
            </span>
          )}
        </section>

        <section className={ws.advice}>
          <p className="label">Requirements</p>
          <span>
            All four from one patient, <Term k="coregistered">co-registered</Term> to one grid (same shape and spacing)
          </span>
          <span>
            <Term k="skullStripped">Skull-stripped</Term>, background exactly 0
          </span>
          <span>
            <span className="mono">1</span> mm isotropic, e.g. <span className="mono">240 × 240 × 155</span>. Other
            spacings run with a warning
          </span>
        </section>

        <p className={ws.footnote}>
          File type, size and the NIfTI header (shape and spacing) are checked in the browser before upload; the server
          checks again. Not a medical device.
        </p>
      </aside>
    </div>
  );
}
