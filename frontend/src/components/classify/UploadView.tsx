"use client";

import Image from "next/image";
import { useId, useRef, useState, type DragEvent } from "react";

import { Button } from "@/components/Button";
import { CheckIcon, ErrorIcon } from "@/components/icons";
import { Term } from "@/components/Term";
import { CLASS_LABELS, type Sample } from "@/lib/api";
import { formatBytes, MAX_IMAGE_MB, MIN_SIDE } from "@/lib/imageCheck";

import styles from "@/components/workspace/Workspace.module.css";
import { SampleNote } from "./SampleNote";
import type { Problem, Selected } from "./types";

type Props = {
  selected: Selected | null;
  problem: Problem | null;
  running: boolean;
  runError: string | null;
  samples: Sample[] | null;
  samplesError: string | null;
  onFile: (file: File) => void;
  onSample: (sample: Sample) => void;
  onClear: () => void;
  onClassify: () => void;
};

export function UploadView(props: Props) {
  const { selected, problem, running, runError, samples, samplesError } = props;
  const inputId = useId();
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) props.onFile(file);
  };

  const info = selected?.info;
  const status = info
    ? `${info.format} · ${info.width} × ${info.height} · ${info.grayscale ? "grayscale" : "colour"}`
    : problem
      ? problem.message
      : "Drop an image here or browse for one";

  return (
    <div className={styles.screen}>
      <main className={styles.main}>
        <div className={styles.intro}>
          <p className="label">Classification · upload</p>
          <h1 className={styles.title}>Name the tumour type in one MRI image</h1>
          <p className={styles.lede}>
            The model looks at a single <Term k="slice">2D slice</Term> and says which of{" "}
            <Term k="classes">four classes</Term> it shows, with a map of where it looked.
          </p>
        </div>

        <section className={styles.tryPanel} aria-labelledby="try-classify">
          <div className={styles.tryText}>
            <p className="label">Try it now</p>
            <h2 id="try-classify" className={styles.tryTitle}>
              Classify an image from the test set
            </h2>
            <p className="caption">
              One click runs the model. These images were never used in training; the second glioma is a{" "}
              <Term k="knownError">known error</Term>.
            </p>
          </div>
          <div className={styles.tryOptions}>
            {samples?.map((sample, i) => (
              <button
                key={sample.name}
                type="button"
                className={styles.tryOption}
                data-primary={i === 0}
                disabled={running}
                onClick={() => props.onSample(sample)}
              >
                <span>{CLASS_LABELS[sample.true_class] ?? sample.true_class}</span>
                <SampleNote sample={sample} name />
              </button>
            ))}
            {!samples && (
              <p className={`caption ${styles.tryPending}`}>
                {samplesError ? `Samples unavailable: ${samplesError}` : "Loading samples…"}
              </p>
            )}
          </div>
        </section>

        <p className={`label ${styles.ownLabel}`}>Or add your own image</p>
        <label
          htmlFor={inputId}
          className={`${styles.drop} ${info ? "" : styles.dropCompact}`}
          data-state={info ? "ready" : dragging ? "dragging" : "empty"}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          {selected && (
            <Image
              className={styles.dropPreview}
              src={selected.previewUrl}
              alt={`Preview of ${selected.info.name}`}
              fill
              unoptimized
            />
          )}
          <span className={styles.dropEmpty} aria-hidden={Boolean(info)}>
            <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
              <path d="M14 19V4M8 10l6-6 6 6M4 23h20" fill="none" stroke="currentColor" strokeWidth="1.4" />
            </svg>
            <span className={styles.dropTitle}>Drop an image here</span>
            <span className={styles.dropHint}>
              JPEG or PNG · up to <span className="mono">{MAX_IMAGE_MB}</span> MB · at least{" "}
              <span className="mono">{MIN_SIDE}</span> px a side
            </span>
          </span>
          <input
            ref={input}
            id={inputId}
            type="file"
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            className={styles.fileInput}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) props.onFile(file);
              event.target.value = "";
            }}
          />
        </label>

        <div className={styles.fileRow} data-state={info ? "ok" : problem ? "bad" : "empty"} aria-live="polite">
          <span className={styles.fileText}>
            <span className={`mono ${styles.fileName}`}>{info?.name ?? problem?.name ?? "No file yet"}</span>
            <span className={styles.fileStatus}>
              {info && <CheckIcon />}
              {problem && !info && <ErrorIcon />}
              <span>{status}</span>
            </span>
          </span>
          <span className={`mono ${styles.fileSize}`}>
            {info ? formatBytes(info.bytes) : problem ? formatBytes(problem.bytes) : ""}
          </span>
          <button
            type="button"
            className={styles.remove}
            aria-label="Remove file"
            onClick={props.onClear}
            disabled={!info && !problem}
          >
            ×
          </button>
        </div>
      </main>

      <aside className={styles.aside} aria-label="Run">
        <section className={styles.stack12}>
          <Button block disabled={!selected || running} onClick={props.onClassify}>
            {running ? "Classifying…" : "Classify"}
          </Button>
          <Button block variant="ghost" onClick={() => input.current?.click()}>
            Browse files
          </Button>
          {runError ? (
            <span className={styles.runError} role="alert">
              <ErrorIcon />
              <span>{runError}</span>
            </span>
          ) : (
            <span className={styles.note}>
              {running
                ? "Running the model…"
                : selected
                  ? "Ready. Classification takes about a second."
                  : "Add a JPEG or PNG slice to continue."}
            </span>
          )}
        </section>

        <section className={styles.advice}>
          <p className="label">What works best</p>
          <span>
            One axial, coronal or sagittal slice, ideally <Term k="sequences">contrast-enhanced T1</Term>
          </span>
          <span>T2, FLAIR and CT images are poorly covered by the training data</span>
        </section>

        <p className={styles.footnote}>
          Type, size and pixel dimensions are checked in the browser; the server checks again. Not a medical device.
        </p>
      </aside>
    </div>
  );
}
