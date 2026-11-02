"use client";

import dynamic from "next/dynamic";
import Image from "next/image";
import { useState } from "react";

import { Button } from "@/components/Button";
import { Term, useDefinition } from "@/components/Term";
import ws from "@/components/workspace/Workspace.module.css";
import { regionColor } from "@/design/tokens";
import { useEntryProgress } from "@/design/useEntryProgress";
import type { LabelLayer, Plane, SegmentationResult, SequenceField } from "@/lib/api";
import { sequenceTerm } from "@/lib/glossary";

import styles from "./Segment.module.css";
import type { Run } from "./types";

// three.js is loaded only when the 3D view is opened, and never on the server.
const Volume3D = dynamic(() => import("./Volume3D"), {
  ssr: false,
  loading: () => (
    <div className={styles.volume3d} data-state="loading">
      <div className={styles.volumeMessage}>
        <p>Loading the 3D viewer…</p>
      </div>
    </div>
  ),
});

/** The plane views plus the 3D surfaces. */
export type View = Plane | "3d";

const PLANES: { key: Plane; name: string; axis: string }[] = [
  { key: "axial", name: "Axial", axis: "z" },
  { key: "coronal", name: "Coronal", axis: "y" },
  { key: "sagittal", name: "Sagittal", axis: "x" },
];
const SEQUENCES: { key: SequenceField; name: string; hint: string }[] = [
  { key: "flair", name: "FLAIR", hint: "swelling (edema) around the tumour shows bright" },
  { key: "t1", name: "T1", hint: "plain anatomy, no contrast agent" },
  { key: "t1ce", name: "T1ce", hint: "actively growing tumour lights up after a contrast injection" },
  { key: "t2", name: "T2", hint: "fluid and swelling show bright" },
];
const LABELS: { key: LabelLayer; value: number; name: string; swatch: string }[] = [
  { key: "edema", value: 2, name: "Edema", swatch: regionColor.WT },
  { key: "necrotic", value: 1, name: "Necrotic core", swatch: regionColor.TC },
  { key: "enhancing", value: 4, name: "Enhancing", swatch: regionColor.ET },
];
const REGIONS = [
  { key: "WT", term: "wt", name: "Whole tumour", swatch: regionColor.WT, large: true },
  { key: "TC", term: "tc", name: "Tumour core", swatch: regionColor.TC, large: false },
  { key: "ET", term: "et", name: "Enhancing tumour", swatch: regionColor.ET, large: false },
] as const;

type Props = { result: SegmentationResult; run: Run; onNewStudy: () => void; initialView?: View };

export function ResultsView({ result, run, onNewStudy, initialView = "axial" }: Props) {
  const [view3d, setView3d] = useState(initialView === "3d");
  const [plane, setPlane] = useState<Plane>(initialView === "3d" ? "axial" : initialView);
  const [sequence, setSequence] = useState<SequenceField>("flair");
  const [shown, setShown] = useState<Record<LabelLayer, boolean>>({ edema: true, necrotic: true, enhancing: true });
  const [opacity, setOpacity] = useState(100);
  const e = useEntryProgress();

  const view = result.slices[plane];
  const planeDef = PLANES.find((p) => p.key === plane)!;
  const seqName = SEQUENCES.find((s) => s.key === sequence)!.name;
  const visible = LABELS.filter((l) => shown[l.key]);
  const t = result.timing_seconds;
  const upload = run.uploadSeconds ?? t.load;
  // A pre-computed sample only took the upload here; its stage timings are from the run at start-up.
  const total = result.precomputed ? upload : upload + t.preprocess + t.inference + t.postprocess_and_encode;
  const spacing = result.spacing_mm.every((s) => Math.abs(s - result.spacing_mm[0]) < 1e-3)
    ? `${result.spacing_mm[0].toFixed(1)} mm`
    : `${result.spacing_mm.map((s) => s.toFixed(2)).join(" × ")} mm`;
  const wt = result.regions.WT.volume_ml || 1;
  const pp = result.post_processing;

  const reportUrl = `${result.report_url}?${new URLSearchParams({
    plane,
    sequence,
    labels: visible.map((l) => l.key).join(","),
    opacity: String(opacity),
  })}`;
  const shownNames = visible.map((l) => l.name.toLowerCase()).join(", ") || "no labels";
  const viewerLabel = view3d
    ? `Interactive 3D view of ${result.study}: the brain surface and the predicted ${shownNames} in red`
    : `${run.sample ? "Real " : ""}${plane} ${seqName} slice of ${result.study} with predicted ${shownNames} in red`;

  return (
    <div className={ws.screen}>
      <main className={`${ws.main} ${ws.resultMain}`}>
        <p className={`mono caption ${ws.context}`}>
          {result.study} · {result.shape.join(" × ")} · {spacing.replace(" mm", "\u00a0mm")} ·{" "}
          {result.precomputed ? "pre-computed" : `${total.toFixed(1)}\u00a0s`}
        </p>
        <div className={`label ${ws.viewerHead}`}>
          {view3d ? (
            <span>3D · brain and tumour surfaces</span>
          ) : (
            <span>
              {planeDef.name} ·{" "}
              <span className="mono">
                {planeDef.axis} {view.index} / {view.count}
              </span>
            </span>
          )}
          <span>
            {view3d ? "Labels" : seqName} · overlay <span className="mono">{opacity}%</span>
          </span>
        </div>

        <div className={ws.viewer} role={view3d ? "group" : "img"} aria-label={viewerLabel}>
          {view3d && (
            <div className={styles.volumeFrame}>
              <Volume3D url={result.mesh_url} visible={shown} opacity={opacity / 100} />
            </div>
          )}
          <div className={ws.viewerInner} style={{ opacity: view3d ? 0 : 1 }} aria-hidden={view3d}>
            {PLANES.map((p) => {
              const layers = result.slices[p.key].layers;
              return (
                <div key={p.key} className={styles.planeStack} style={{ opacity: p.key === plane ? 1 : 0 }} aria-hidden={p.key !== plane}>
                  {SEQUENCES.map((s) => (
                    <Image
                      key={s.key}
                      className={ws.layer}
                      src={layers[s.key]}
                      alt=""
                      fill
                      unoptimized
                      style={{ opacity: s.key === sequence ? 1 : 0 }}
                    />
                  ))}
                  <div className={styles.planeStack} style={{ opacity: opacity / 100 }}>
                    {LABELS.map((l) => (
                      <Image
                        key={l.key}
                        className={ws.layer}
                        src={layers[l.key]}
                        alt=""
                        fill
                        unoptimized
                        style={{ opacity: shown[l.key] ? 1 : 0 }}
                      />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className={ws.toolbar} role="toolbar" aria-label="Viewer controls">
          <div className={ws.modes} role="group" aria-label="Plane">
            {PLANES.map((p) => (
              <button
                key={p.key}
                type="button"
                className={`${ws.mode} ${styles.planeButton}`}
                aria-pressed={!view3d && plane === p.key}
                onClick={() => {
                  setPlane(p.key);
                  setView3d(false);
                }}
              >
                {p.name}
              </button>
            ))}
            <button
              type="button"
              className={`${ws.mode} ${styles.planeButton}`}
              aria-pressed={view3d}
              onClick={() => setView3d(true)}
            >
              3D
            </button>
          </div>
          <div className={ws.modes} role="group" aria-label="Base sequence">
            {SEQUENCES.map((s) => (
              <SequenceButton
                key={s.key}
                name={s.name}
                pressed={sequence === s.key}
                disabled={view3d}
                onSelect={() => setSequence(s.key)}
              />
            ))}
            <span className={styles.groupTerm}>
              <Term k="sequences" />
            </span>
          </div>
          <label className={`label ${ws.heat} ${styles.opacity}`}>
            <span>Opacity</span>
            <input
              type="range"
              min={0}
              max={100}
              value={opacity}
              onChange={(event) => setOpacity(Number(event.target.value))}
              aria-valuetext={`${opacity}%`}
            />
          </label>
        </div>
        {!view3d && (
          <p className={`caption ${styles.seqHint}`} aria-live="polite">
            Showing <span className={styles.seqHintName}>{seqName}</span>: {SEQUENCES.find((s) => s.key === sequence)!.hint}.
          </p>
        )}
        <div className={styles.legend} role="group" aria-label="Labels shown">
          {LABELS.map((l) => (
            <button
              key={l.key}
              type="button"
              className={styles.legendItem}
              aria-pressed={shown[l.key]}
              onClick={() => setShown((s) => ({ ...s, [l.key]: !s[l.key] }))}
            >
              <span className={styles.swatch} style={{ background: l.swatch }} />
              <span className="mono">{l.value}</span> {l.name}
            </button>
          ))}
          <span className={styles.legendTerm}>
            <Term k="labels" />
          </span>
        </div>

        {/* Narrow screens: the measurements rail is below the viewer, so keep the result and a way
            to start again in view. */}
        <div className={ws.mobileResult}>
          <span className={ws.mobileResultText}>
            <span>{result.study}</span>
            <span className="mono caption">WT {result.regions.WT.volume_ml.toFixed(2)} mL</span>
          </span>
          <Button variant="ghost" onClick={onNewStudy} className={ws.mobileReset}>
            New study
          </Button>
        </div>
      </main>

      <aside className={ws.aside} aria-label="Measurements">
        <section className={styles.regions}>
          {REGIONS.map((r) => {
            const region = result.regions[r.key];
            return (
              <div key={r.key} className={styles.region}>
                <span className={`label ${styles.regionLabel}`}>
                  <span className={styles.swatchSmall} style={{ background: r.swatch }} />
                  <Term k={r.term}>
                    {r.key} · {r.name}
                  </Term>
                </span>
                <span className={`mono ${styles.volume}`} data-large={r.large}>
                  {(region.volume_ml * e).toFixed(2)}
                  <span className={styles.volumeUnit}> mL</span>
                </span>
                <div className={ws.track} aria-hidden="true">
                  <div className={styles.shareFill} style={{ width: `${(region.volume_ml / wt) * 100 * e}%` }} />
                </div>
                <span className="mono caption">
                  <Term k="agreement">agreement</Term> {region.model_agreement.toFixed(3)} ·{" "}
                  <Term k="meanP">mean p</Term>{" "}
                  {region.mean_probability === null ? "–" : region.mean_probability.toFixed(3)}
                </span>
              </div>
            );
          })}
          {result.warnings.map((w) => (
            <p key={w} className={`caption ${ws.warning}`}>
              {w}
            </p>
          ))}
        </section>

        <section className={ws.stack10}>
          <a className={ws.download} href={result.mask_url} download>
            Download label map (.nii.gz)
          </a>
          {view3d ? (
            <a className={styles.ghostLink} href={result.mesh_url} download={`${result.study}_surfaces.glb`}>
              Download 3D surfaces (.glb)
            </a>
          ) : (
            <a className={styles.ghostLink} href={reportUrl} download>
              Download {plane} overlay (PNG)
            </a>
          )}
          <span className="caption">
            The <Term k="mask">label map</Term> is the full 3D result for viewers such as ITK-SNAP.{" "}
            {view3d ? (
              <>
                The <Term k="glb">.glb</Term> holds the surfaces shown here, for any glTF viewer or 3D tool.
              </>
            ) : (
              "The PNG is the slice on screen with the overlay burned in, for slides and reports."
            )}
          </span>
          <Button variant="ghost" block onClick={onNewStudy}>
            New study
          </Button>
        </section>

        <section className={ws.details}>
          <details>
            <summary className="label">Inputs and checks</summary>
            <p className={`mono caption ${ws.modelLines}`}>
              {SEQUENCES.map((s) => (
                <span key={s.key}>
                  {s.name.toLowerCase()} · {result.files[s.key] ?? "–"}
                  <br />
                </span>
              ))}
              same grid · same spacing · {result.warnings.length ? `${result.warnings.length} warning(s)` : "no warnings"}
            </p>
          </details>
          <details>
            <summary className="label">ET post-processing</summary>
            <p className={`caption ${styles.detailText}`}>
              {pp.et_components_removed} of {pp.et_components_predicted} enhancing component
              {pp.et_components_predicted === 1 ? "" : "s"} removed ({pp.et_voxels_removed} voxel
              {pp.et_voxels_removed === 1 ? "" : "s"}). Components under {pp.et_min_component_voxels} voxels are dropped;
              the threshold was tuned on out-of-fold validation data.
            </p>
          </details>
          <details>
            <summary className="label">
              Timing · <span className="mono">{total.toFixed(2)} s</span>
            </summary>
            <p className={`mono caption ${ws.modelLines}`}>
              {run.files ? "upload and check" : "request"} {upload.toFixed(2)}
              {result.precomputed ? (
                <>
                  <br />
                  pre-computed at start-up:{" "}
                </>
              ) : (
                " · "
              )}
              preprocess {t.preprocess.toFixed(2)}
              <br />
              inference, {result.model.folds.length} models {t.inference.toFixed(2)} · encode{" "}
              {t.postprocess_and_encode.toFixed(2)}
            </p>
          </details>
          <details>
            <summary className="label">Ensemble</summary>
            <p className={`mono caption ${ws.modelLines}`}>
              {result.model.folds.length} folds · CV mean Dice
              <br />
              {Object.values(result.model.cross_validation_mean_dice)
                .map((d) => d.toFixed(3))
                .join(" · ")}
            </p>
          </details>
        </section>

        <p className={ws.footnote}>
          {run.sample ? "Real ensemble output for a held-out test patient." : "Ensemble output for your upload."} WT =
          labels 1+2+4, TC = 1+4, ET = 4. Bars show each region&apos;s share of the whole tumour. Agreement is the mean
          Dice between each fold model and the ensemble: consistency, not accuracy. Not a medical device.
        </p>
      </aside>
    </div>
  );
}

/**
 * A sequence toggle that also shows what the sequence is: on hover, on keyboard focus, and on a
 * tap, which selects the sequence and shows its definition until the next tap elsewhere.
 */
function SequenceButton({ name, pressed, disabled, onSelect }: { name: string; pressed: boolean; disabled: boolean; onSelect: () => void }) {
  const { id, setAnchor, open, show, hide, popover } = useDefinition<HTMLButtonElement>(sequenceTerm(name));
  return (
    <>
      <button
        ref={setAnchor}
        type="button"
        className={`${ws.mode} ${styles.seqButton}`}
        aria-pressed={pressed}
        aria-describedby={open ? id : undefined}
        disabled={disabled}
        title={disabled ? "Sequences apply to the slice views" : undefined}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        onPointerUp={(event) => event.pointerType !== "mouse" && show()}
        onClick={onSelect}
      >
        {name}
      </button>
      {popover}
    </>
  );
}
