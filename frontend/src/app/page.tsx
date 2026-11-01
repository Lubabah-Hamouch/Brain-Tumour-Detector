import Link from "next/link";
import { Suspense } from "react";

import { ButtonLink } from "@/components/ButtonLink";
import { HeroVisual } from "@/components/HeroVisual";
import { HeroMetric, MetricRow } from "@/components/Metric";
import { ApiError, ApiUnavailable, getResultsSummary } from "@/lib/api";

import styles from "./page.module.css";

const INDEX = [
  { n: "01", href: "/segment", title: "Segment a study", body: "FLAIR, T1, T1ce and T2 in, label map and region volumes out. About 6 s on a GPU." },
  { n: "02", href: "/classify", title: "Classify a slice", body: "One 2D image in, tumour type, probabilities and a Grad-CAM map out." },
  { n: "03", href: "/model-card", title: "Read the model card", body: "Test results, the image-format shortcut found and mitigated, and the limitations." },
];

const PIPELINE = {
  columns: ["Data", "Split", "Train", "Evaluate", "Serve"],
  rows: [
    {
      name: "Segment",
      cells: [
        "1,251 BraTS 2021 patients, four sequences each",
        "189-patient test set, 5 stratified patient-level folds",
        "3D U-Net, Dice + BCE per region, deep supervision",
        "Sliding window, 5-fold ensemble, ET clean-up tuned on validation",
        "NIfTI label map in the upload's own grid",
      ],
    },
    {
      name: "Classify",
      cells: [
        "6,951 slices after de-duplication and cleaning",
        "70 / 15 / 15, near-duplicate clusters kept in one split",
        "EfficientNetV2-S, crop + resolution augmentation",
        "Test set plus a format stress test for shortcut learning",
        "Probabilities and a Grad-CAM map",
      ],
    },
  ],
};

export default function OverviewPage() {
  return (
    <main>
      <section className={styles.hero}>
        <div className={styles.heroText}>
          <div className={styles.stack28}>
            <p className="label">Research demonstration · not for clinical use</p>
            <h1 className={styles.headline}>Tumour segmentation and tumour-type classification for brain MRI</h1>
          </div>
          <div className={styles.stack32}>
            <p className={styles.lead}>
              A five-model 3D U-Net ensemble outlines glioma sub-regions in multimodal MRI. A fine-tuned EfficientNetV2-S
              names the tumour type in a single 2D slice. Both are scored on test sets held out from every training and
              selection decision.
            </p>
            <div className={styles.actions}>
              <ButtonLink href="/segment">Segment a study</ButtonLink>
              <ButtonLink href="/classify" variant="ghost">
                Classify a slice
              </ButtonLink>
            </div>
          </div>
        </div>
        <HeroVisual />
      </section>

      <Suspense fallback={<ResultsPlaceholder />}>
        <HeadlineResults />
      </Suspense>

      <section aria-labelledby="index-heading" className={styles.section}>
        <h2 id="index-heading" className={`label ${styles.sectionLabel}`}>
          Index
        </h2>
        {INDEX.map((item) => (
          <Link key={item.n} href={item.href} className={styles.indexRow}>
            <span className={`mono ${styles.indexNumber}`}>{item.n}</span>
            <span className={styles.indexTitle}>
              <span className="u">{item.title}</span>
            </span>
            <span className={styles.indexBody}>{item.body}</span>
          </Link>
        ))}
        <div className={styles.rule} />
      </section>

      <section aria-labelledby="pipeline-heading" className={styles.section}>
        <h2 id="pipeline-heading" className={`label ${styles.sectionLabel}`}>
          Pipeline
        </h2>
        <div className={styles.tableScroll}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th scope="col" />
                {PIPELINE.columns.map((c) => (
                  <th key={c} scope="col" className="label">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {PIPELINE.rows.map((row) => (
                <tr key={row.name}>
                  <th scope="row" className={`label ${styles.rowName}`}>
                    {row.name}
                  </th>
                  {row.cells.map((cell) => (
                    <td key={cell}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

/** Headline test results, read from the API (which reads reports/results/) at request time. */
async function HeadlineResults() {
  let summary;
  try {
    summary = await getResultsSummary();
  } catch (error) {
    if (!(error instanceof ApiUnavailable || error instanceof ApiError)) throw error;
    return (
      <section aria-label="Headline results" className={styles.results}>
        <p className={styles.unavailable}>
          Test results are unavailable: {error.message} Start the API with{" "}
          <code className="mono">uvicorn backend.main:app --port 8010</code> and reload.
        </p>
      </section>
    );
  }
  const { segmentation: seg, classification: clf } = summary;
  const [lo, hi] = clf.accuracy_ci95;
  return (
    <section aria-label="Headline results" className={styles.results}>
      <div className={styles.resultColumn}>
        <HeroMetric label="Segmentation · mean Dice" value={seg.dice.mean} decimals={3} />
        <div className={styles.rows}>
          <MetricRow name="WT" value={seg.dice.WT} decimals={3} />
          <MetricRow name="TC" value={seg.dice.TC} decimals={3} />
          <MetricRow name="ET" value={seg.dice.ET} decimals={3} />
        </div>
        <p className="caption">
          5-fold ensemble · <span className="mono">{seg.test_patients}</span> held-out patients
        </p>
      </div>
      <div className={styles.resultColumn}>
        <HeroMetric label="Classification · accuracy" value={clf.accuracy} scale={100} decimals={1} unit="%" />
        <dl className={styles.facts}>
          <dt>95% confidence interval</dt>
          <dd className="mono">
            {(lo * 100).toFixed(1)}–{(hi * 100).toFixed(1)}
          </dd>
          <dt>Macro F1</dt>
          <dd className="mono">{clf.macro_f1.toFixed(3)}</dd>
          {clf.accuracy_at_128px !== null && (
            <>
              <dt>Test images shrunk to 128 px</dt>
              <dd className="mono">{(clf.accuracy_at_128px * 100).toFixed(1)}%</dd>
            </>
          )}
        </dl>
        <p className="caption">
          <span className="mono">{clf.test_images.toLocaleString("en")}</span> held-out images · 4 classes
        </p>
      </div>
    </section>
  );
}

function ResultsPlaceholder() {
  return (
    <section aria-label="Headline results" aria-busy="true" className={styles.results}>
      <div className={styles.resultColumn}>
        <span className="label">Segmentation · mean Dice</span>
        <span className={styles.placeholderNumber} />
      </div>
      <div className={styles.resultColumn}>
        <span className="label">Classification · accuracy</span>
        <span className={styles.placeholderNumber} />
      </div>
    </section>
  );
}
