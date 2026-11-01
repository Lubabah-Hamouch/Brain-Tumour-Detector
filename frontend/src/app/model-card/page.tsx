import type { Metadata } from "next";
import { Suspense, type ReactNode } from "react";

import { StressChart } from "@/components/modelcard/StressChart";
import { StressHeadline } from "@/components/modelcard/StressHeadline";
import styles from "@/components/modelcard/ModelCard.module.css";
import { ApiError, ApiUnavailable, getModelCard, type ModelCard, type PairedComparison } from "@/lib/api";

export const metadata: Metadata = { title: "Model card" };

export default function ModelCardPage() {
  return (
    <main>
      <Suspense fallback={<Pending />}>
        <ModelCardContent />
      </Suspense>
    </main>
  );
}

function Pending() {
  return (
    <section className={styles.top} aria-busy="true">
      <div className={styles.topMain}>
        <p className="label">Model card · classifier · format stress test</p>
        <span className={styles.heroPlaceholder} />
      </div>
    </section>
  );
}

// ---- Formatting: every number below comes from the API, which reads the results files. ----

const MINUS = "−";
const pct = (v: number, d = 2) => `${(v * 100).toFixed(d)}%`;
const dice = (v: number) => v.toFixed(3);
/** Signed difference; 4 decimals when it would otherwise round to zero. */
function signed(v: number) {
  const digits = v !== 0 && Math.abs(v) < 0.0005 ? 4 : 3;
  const text = Math.abs(v).toFixed(digits);
  return `${v < 0 ? MINUS : "+"}${text}`;
}
const millions = (n: number | null) => (n === null ? "–" : `${(n / 1e6).toFixed(1)} M`);
const pValue = (p: number) => (p >= 0.1 ? p.toFixed(1) : p.toPrecision(2));
const significant = (c: PairedComparison) => c.ci95[0] > 0 || c.ci95[1] < 0;

async function ModelCardContent() {
  let card: ModelCard;
  try {
    card = await getModelCard();
  } catch (error) {
    if (!(error instanceof ApiUnavailable || error instanceof ApiError)) throw error;
    return (
      <section className={styles.top}>
        <p className={styles.unavailable}>
          The model card is unavailable: {error.message} Start the API with{" "}
          <code className="mono">uvicorn backend.main:app --port 8010</code> and reload.
        </p>
      </section>
    );
  }

  const { headline: h, specification: spec, segmentation: seg, classification: clf } = card;
  const segSpec = spec.segmentation;
  const clfSpec = spec.classification;
  const patch = segSpec.patch_size;
  const patchText =
    patch === null ? "–" : typeof patch === "number" || patch.every((p) => p === patch[0]) ? `${[patch].flat()[0]}³` : patch.join(" × ");
  const split = clfSpec.split;
  const bestFold = seg.single_best_fold.replace(/^fold/, "");
  const ens = seg.ensemble_vs_single_best;
  const cleanup = seg.et_cleanup_on_ensemble;
  const noEt = seg.patients_without_et;
  const scored = seg.patients_without_et_scored_1;

  const specRows: [string, string, string][] = [
    [
      "Task",
      "Outline whole tumour, tumour core and enhancing tumour in 3D",
      "Glioma, meningioma, no tumour or pituitary from one 2D slice",
    ],
    [
      "Data",
      `BraTS 2021, ${segSpec.patients.toLocaleString("en")} patients, FLAIR + T1 + T1ce + T2`,
      `Brain Tumor MRI dataset, ${clfSpec.images.toLocaleString("en")} slices after cleaning`,
    ],
    [
      "Split",
      `${segSpec.test_patients} test patients · ${segSpec.folds} stratified patient-level folds`,
      `${Math.round(split.train * 100)} / ${Math.round(split.val * 100)} / ${Math.round(split.test * 100)}, near-duplicate clusters kept in one split`,
    ],
    [
      "Model",
      `3D U-Net, ${millions(segSpec.parameters)} parameters, ${segSpec.folds}-fold ensemble`,
      `EfficientNetV2-S, ${millions(clfSpec.parameters)} parameters, ImageNet start`,
    ],
    [
      "Inference",
      `${patchText} sliding window, Gaussian blending, ET components < ${segSpec.et_min_voxels} voxels removed`,
      `Grayscale, letterbox to ${clfSpec.image_size ?? "–"} px, Grad-CAM on the predicted class`,
    ],
  ];

  const sameAccuracy = h.served_accuracy === h.original_accuracy;

  return (
    <>
      <section className={styles.top}>
        <div className={styles.topMain}>
          <h1 className="label">Model card · classifier · format stress test</h1>
          <StressHeadline served={h.served_accuracy_at_128px} original={h.original_accuracy_at_128px} />
        </div>
        <p className={styles.topNote}>
          {sameAccuracy
            ? `On natural test images both models score ${pct(h.served_accuracy)}`
            : `On natural test images the served model scores ${pct(h.served_accuracy)} and the original ${pct(h.original_accuracy)}`}
          {h.mcnemar_p !== null &&
            ` and are ${h.mcnemar_p >= 0.05 ? "statistically indistinguishable" : "significantly different"} (McNemar p = ${pValue(h.mcnemar_p)})`}
          . The difference only appears when file format stops matching the source each class came from, which is why
          the served model was chosen. Every number on this page comes from held-out test data.
        </p>
      </section>

      <div className={styles.sections}>
        <section aria-labelledby="mc-spec" className={styles.specSection}>
          <h2 id="mc-spec" className="label">
            Specification
          </h2>
          <div className={styles.scroll}>
            <table className={styles.spec}>
              <thead>
                <tr>
                  <th scope="col" />
                  <th scope="col">Segmentation</th>
                  <th scope="col">Classification</th>
                </tr>
              </thead>
              <tbody>
                {specRows.map(([name, a, b]) => (
                  <tr key={name}>
                    <th scope="row">{name}</th>
                    <td>{a}</td>
                    <td>{b}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section aria-labelledby="mc-seg" className={styles.block}>
          <div className={styles.blockHead}>
            <h2 id="mc-seg" className={styles.h2}>
              Segmentation
            </h2>
            <span className="label">{seg.test_patients} test patients · Dice</span>
          </div>
          <div className={styles.scroll}>
            <table className={`${styles.data} ${styles.segTable}`}>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">ET clean-up</th>
                  <th scope="col">WT</th>
                  <th scope="col">TC</th>
                  <th scope="col">ET</th>
                  <th scope="col">Mean</th>
                </tr>
              </thead>
              <tbody>
                {seg.rows.map((r) => (
                  <tr key={`${r.model}-${r.post_processed}`} data-served={r.served}>
                    <th scope="row">
                      {r.model === "ensemble" ? `${segSpec.folds}-fold ensemble` : `Best single fold (${bestFold})`}
                      {r.served && " · served"}
                    </th>
                    <td>{r.post_processed ? "yes" : "no"}</td>
                    <td>{dice(r.WT)}</td>
                    <td>{dice(r.TC)}</td>
                    <td>{dice(r.ET)}</td>
                    <td className={styles.total}>{dice(r.mean)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className={styles.findings}>
            <Finding value={signed(ens.difference)} title="Ensemble vs best single fold">
              95% CI <span className="mono">{signed(ens.ci95[0])}</span> to <span className="mono">{signed(ens.ci95[1])}</span>
              {significant(ens) ? "" : ", not significant"} · better for {ens.patients_better} of {seg.test_patients} patients
            </Finding>
            <Finding value={signed(cleanup.difference)} title="ET clean-up">
              95% CI <span className="mono">{signed(cleanup.ci95[0])}</span> to{" "}
              <span className="mono">{signed(cleanup.ci95[1])}</span>
              {significant(cleanup) ? "" : ", not significant"} · the gain comes from the {noEt} patients without enhancing
              tumour: {scored.raw} of {noEt} have no false ET before clean-up,{" "}
              {scored.post_processed === noEt ? "all" : scored.post_processed} after
            </Finding>
            <Finding
              value={
                <>
                  {dice(seg.cross_validation.mean)} <span className={styles.findingSpread}>± {dice(seg.cross_validation.std)}</span>
                </>
              }
              title={`${seg.cross_validation.folds}-fold cross-validation`}
            >
              Mean Dice across folds (± standard deviation), each on the validation fold that chose its checkpoint
            </Finding>
          </div>
        </section>

        <section aria-labelledby="mc-clf" className={styles.block}>
          <div className={styles.blockHead}>
            <h2 id="mc-clf" className={styles.h2}>
              Classification
            </h2>
            <span className="label">
              {clf.test_images.toLocaleString("en")} test images · {clf.models.length} models considered
            </span>
          </div>
          <div className={styles.scroll}>
            <table className={`${styles.data} ${styles.clfTable}`}>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Training change</th>
                  <th scope="col">Accuracy</th>
                  <th scope="col">Macro F1</th>
                  <th scope="col">At 128 px</th>
                </tr>
              </thead>
              <tbody>
                {clf.models.map((m) => (
                  <tr key={m.key} data-served={m.served}>
                    <th scope="row">
                      {m.key} · {m.name}
                      {m.served && " · served"}
                    </th>
                    <td className={styles.text}>{m.change}</td>
                    <td>{pct(m.accuracy)}</td>
                    <td>{m.macro_f1.toFixed(3)}</td>
                    <td className={styles.total}>{pct(m.accuracy_at_128px)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <figure className={styles.figure}>
            <figcaption className={styles.figcaption}>
              <span className="label">Format stress test</span>
              <span className={styles.figTitle}>
                Tumour images (of {card.stress_test.tumour_images}) called “no tumour” when only the file size changes
              </span>
            </figcaption>
            <StressChart data={card.stress_test} />
            <p className={styles.figNote}>
              The stress test uses synthetic down-sampling, which is related to the training augmentation; robustness to
              images from a genuinely different source still needs an external dataset.
            </p>
          </figure>
        </section>

        <section aria-labelledby="mc-lim" className={styles.limitsSection}>
          <h2 id="mc-lim" className={styles.h2}>
            Limitations
          </h2>
          <ol className={styles.limits}>
            {card.limitations.map((l, i) => (
              <li key={l.title}>
                <span className="mono caption">{String(i + 1).padStart(2, "0")}</span>
                <span className={styles.limitTitle}>{l.title}</span>
                <span />
                <span className={styles.limitBody}>{l.body}</span>
              </li>
            ))}
          </ol>
        </section>

        <section aria-label="Sources" className={styles.sources}>
          <p className="caption">
            Full tables and methods: <span className="mono">reports/final_results.md</span>. This page reads:{" "}
            {card.sources.map((s, i) => (
              <span key={s}>
                {i > 0 && " · "}
                <span className="mono">{s}</span>
              </span>
            ))}
            .
          </p>
        </section>
      </div>
    </>
  );
}

function Finding({ value, title, children }: { value: ReactNode; title: string; children: ReactNode }) {
  return (
    <div className={styles.finding}>
      <span className={`mono ${styles.findingValue}`}>{value}</span>
      <span className={styles.findingTitle}>{title}</span>
      <span className="caption">{children}</span>
    </div>
  );
}
