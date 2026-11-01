"use client";

import Image from "next/image";
import { useId, useState } from "react";

import { useEntryProgress } from "@/design/useEntryProgress";
import { CLASS_LABELS, type ClassificationResult, type ResultsSummary, type Sample } from "@/lib/api";

import styles from "@/components/workspace/Workspace.module.css";
import { SampleNote } from "./SampleNote";
import type { Selected } from "./types";

const MODES = ["Input", "Heatmap", "Overlay"] as const;
type Mode = (typeof MODES)[number];

type Props = {
  result: ClassificationResult;
  selected: Selected;
  summary: ResultsSummary | null;
  samples: Sample[] | null;
  running: boolean;
  onFile: (file: File) => void;
  onSample: (sample: Sample) => void;
  onReset: () => void;
};

const label = (cls: string) => CLASS_LABELS[cls] ?? cls;

/** "Runner-up: glioma at 0.169." plus how often that pair is confused on the test set. */
function runnerUpNote(result: ClassificationResult, summary: ResultsSummary | null) {
  const ranked = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  const [second, p] = ranked[1];
  const sentence = `Runner-up: ${label(second).toLowerCase()} at ${p.toFixed(3)}.`;
  const clf = summary?.classification;
  const pair = clf?.confusions.find((c) => c.classes.includes(result.predicted_class) && c.classes.includes(second));
  if (!clf || !pair) return sentence;
  const top = Math.max(...clf.confusions.map((c) => c.errors));
  const tied = clf.confusions.filter((c) => c.errors === top).length;
  const rank =
    pair.errors === top ? (tied > 1 ? `, joint most frequent with ${tied - 1} other pair${tied > 2 ? "s" : ""}` : ", the most frequent pair") : "";
  return `${sentence} ${label(pair.classes[0])} and ${label(pair.classes[1]).toLowerCase()} account for ${pair.errors} of the ${clf.test_errors} test-set errors${rank}.`;
}

export function ResultView({ result, selected, summary, samples, running, onFile, onSample, onReset }: Props) {
  const [mode, setMode] = useState<Mode>("Overlay");
  const [opacity, setOpacity] = useState(85);
  const uploadId = useId();
  const e = useEntryProgress();

  const ranked = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  const { info, sample } = selected;
  const size = result.letterbox.size;
  const baseOpacity = mode === "Heatmap" ? 0.15 : 1;
  const heatOpacity = mode === "Input" ? 0 : opacity / 100;
  const predicted = label(result.predicted_class);

  return (
    <div className={styles.screen} aria-busy={running}>
      <main className={`${styles.main} ${styles.resultMain}`}>
        <p className={`mono caption ${styles.context}`}>
          {info.name} · {info.width} × {info.height}
          {sample ? " · held-out test" : ""}
        </p>
        <div className={`label ${styles.viewerHead}`}>
          <span>
            Model input · <span className="mono">{size} × {size}</span> letterbox
          </span>
          <span>Grad-CAM · {mode}</span>
        </div>

        <div
          className={styles.viewer}
          role="img"
          aria-label={`${info.name} as the model sees it, with the Grad-CAM heatmap for ${predicted.toLowerCase()} in red`}
        >
          <div className={styles.viewerInner}>
            <Image
              className={`${styles.layer} ${styles.base}`}
              src={`data:image/png;base64,${result.model_input_png}`}
              alt=""
              fill
              unoptimized
              style={{ opacity: baseOpacity }}
            />
            <Image
              className={styles.layer}
              src={`data:image/png;base64,${result.gradcam.heat_png}`}
              alt=""
              fill
              unoptimized
              style={{ opacity: heatOpacity }}
            />
          </div>
        </div>

        <div className={styles.toolbar} role="toolbar" aria-label="View">
          <div className={styles.modes} role="group" aria-label="View mode">
            {MODES.map((m) => (
              <button key={m} type="button" className={styles.mode} aria-pressed={mode === m} onClick={() => setMode(m)}>
                {m}
              </button>
            ))}
          </div>
          <label className={`label ${styles.heat}`}>
            <span>Heat</span>
            <input
              type="range"
              min={0}
              max={100}
              value={opacity}
              onChange={(event) => setOpacity(Number(event.target.value))}
              disabled={mode === "Input"}
              aria-valuetext={`${opacity}%`}
            />
            <span className={`mono ${styles.heatValue}`}>{opacity}%</span>
          </label>
          <span className="caption">From a 12 × 12 feature map · indicative only</span>
        </div>
      </main>

      <aside className={styles.aside} aria-label="Prediction">
        <section className={styles.prediction}>
          <p className="label">Prediction</p>
          <h1 className={styles.predicted}>{predicted}</h1>
          <span className={`mono ${styles.confidence}`}>{(result.confidence * e).toFixed(3)}</span>
          <div className={styles.track} aria-hidden="true">
            <div className={styles.fillFinding} style={{ width: `${result.confidence * 100 * e}%` }} />
          </div>
          <span className="caption">Probability of the predicted class</span>
        </section>

        <section>
          <p className={`label ${styles.sectionLabel}`}>All classes</p>
          <ul className={styles.probs}>
            {ranked.map(([cls, p]) => {
              const lead = cls === result.predicted_class;
              return (
                <li key={cls} className={styles.prob} data-lead={lead}>
                  <span>{label(cls)}</span>
                  <div className={styles.track} aria-hidden="true">
                    <div
                      className={lead ? styles.fillFinding : styles.fillDim}
                      style={{ width: `${Math.max(p * 100 * e, 0.5)}%` }}
                    />
                  </div>
                  <span className="mono">{(p * e).toFixed(4)}</span>
                </li>
              );
            })}
          </ul>
          <p className={`caption ${styles.runnerUp}`}>{runnerUpNote(result, summary)}</p>
          {result.warnings.map((w) => (
            <p key={w} className={`caption ${styles.warning}`}>
              {w}
            </p>
          ))}
        </section>

        <section className={styles.stack10}>
          <a className={styles.download} href={result.report_url} download>
            Download result (PNG)
          </a>
          <span className="caption">One image with the slice, the Grad-CAM overlay, the prediction and all four probabilities.</span>
          <button type="button" className={styles.textButton} onClick={onReset}>
            <span className="u">Classify another image</span>
          </button>
        </section>

        <section className={styles.details}>
          <details>
            <summary className="label">Upload a slice</summary>
            <label htmlFor={uploadId} className="caption">
              JPEG or PNG, up to 20 MB. Classified as soon as it is chosen.
            </label>
            <input
              id={uploadId}
              type="file"
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              className={styles.inlineFile}
              disabled={running}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) onFile(file);
                event.target.value = "";
              }}
            />
          </details>
          <details>
            <summary className="label">Test-set samples</summary>
            <ul className={styles.sampleList}>
              {samples?.map((s) => (
                <li key={s.name}>
                  <button
                    type="button"
                    className={styles.sampleInline}
                    aria-current={s.name === sample?.name ? "true" : undefined}
                    disabled={running}
                    onClick={() => onSample(s)}
                  >
                    <span className="mono">{s.name}</span>
                    <SampleNote sample={s} suffix={s.name === sample?.name ? "shown" : undefined} />
                  </button>
                </li>
              ))}
            </ul>
          </details>
          <details>
            <summary className="label">Model</summary>
            <p className={`mono caption ${styles.modelLines}`}>
              EfficientNetV2-S · epoch {result.model.checkpoint_epoch}
              <br />
              {result.model.checkpoint}
              {summary && (
                <>
                  <br />
                  test accuracy {(summary.classification.accuracy * 100).toFixed(1)}% (
                  {(summary.classification.accuracy_ci95[0] * 100).toFixed(1)}–
                  {(summary.classification.accuracy_ci95[1] * 100).toFixed(1)})
                  {summary.classification.accuracy_at_128px !== null && (
                    <>
                      <br />
                      {(summary.classification.accuracy_at_128px * 100).toFixed(1)}% with images shrunk to 128 px
                    </>
                  )}
                </>
              )}
            </p>
          </details>
        </section>

        <p className={styles.footnote}>
          {sample ? "Real model output for a held-out test image." : "Model output for your upload."} Trained on single
          slices, mostly contrast-enhanced T1; T2, FLAIR and CT are poorly covered. Not a medical device.
        </p>
      </aside>
    </div>
  );
}
