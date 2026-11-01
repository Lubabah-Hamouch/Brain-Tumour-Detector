"use client";

import Image from "next/image";
import { useId, useRef, useState, type DragEvent } from "react";

import { Button } from "@/components/Button";
import type { Sample } from "@/lib/api";
import { formatBytes, MAX_IMAGE_MB, MIN_SIDE } from "@/lib/imageCheck";

import styles from "./Classify.module.css";
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
      : "Drop an image or choose a sample";

  return (
    <div className={styles.screen}>
      <main className={styles.main}>
        <div className={styles.intro}>
          <p className="label">Classification · upload</p>
          <h1 className={styles.title}>Add one 2D MRI slice</h1>
        </div>

        <label
          htmlFor={inputId}
          className={styles.drop}
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
              {info && (
                <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
                  <path d="M2.5 7.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="1.5" />
                </svg>
              )}
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

        <section>
          <p className={`label ${styles.sectionLabel}`}>Or use a test-set sample</p>
          {samples?.map((sample) => (
            <button
              key={sample.name}
              type="button"
              className={styles.sample}
              aria-pressed={selected?.sample?.name === sample.name}
              onClick={() => props.onSample(sample)}
            >
              <span className="mono">{sample.name}</span>
              <SampleNote sample={sample} />
            </button>
          ))}
          {!samples && (
            <p className={styles.samplePending}>{samplesError ? `Samples unavailable: ${samplesError}` : "Loading samples…"}</p>
          )}
          <div className={styles.rule} />
        </section>

        <section className={styles.advice}>
          <p className="label">What works best</p>
          <span>One axial, coronal or sagittal slice, ideally contrast-enhanced T1</span>
          <span>T2, FLAIR and CT images are poorly covered by the training data</span>
        </section>

        <p className={styles.footnote}>
          Type, size and pixel dimensions are checked in the browser; the server checks again. Not a medical device.
        </p>
      </aside>
    </div>
  );
}

export function ErrorIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
      <circle cx="7" cy="7" r="6" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M7 3.8v4M7 9.6v0.6" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}
