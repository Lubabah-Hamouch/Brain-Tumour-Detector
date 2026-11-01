"use client";

import { useEntryProgress } from "@/design/useEntryProgress";

import styles from "./ModelCard.module.css";

/** The served classifier's accuracy on 128 px images, counting up, against the original model's. */
export function StressHeadline({ served, original }: { served: number; original: number }) {
  const e = useEntryProgress();
  const pct = (v: number, d: number) => (v * 100).toFixed(d);
  return (
    <>
      <span className={`mono ${styles.hero}`} aria-label={`${pct(served, 1)}%`}>
        <span aria-hidden="true">
          {pct(served * e, 1)}
          <span className={styles.heroUnit}>%</span>
        </span>
      </span>
      <p className={styles.heroClaim}>
        Accuracy when every test image is shrunk to 128 px, a file size the original model had learned to read as “no
        tumour”.
      </p>
      <div className={styles.compare}>
        <span className={styles.compareServed}>Served model</span>
        <Track fraction={served * e} />
        <span className={`mono ${styles.compareValue}`}>{pct(served * e, 2)}%</span>
        <span className={styles.compareOther}>Original model</span>
        <Track fraction={original * e} dim />
        <span className={`mono ${styles.compareValue} ${styles.compareOther}`}>{pct(original * e, 2)}%</span>
      </div>
    </>
  );
}

function Track({ fraction, dim = false }: { fraction: number; dim?: boolean }) {
  return (
    <div className={styles.track} aria-hidden="true">
      <div className={dim ? styles.trackFillDim : styles.trackFill} style={{ width: `${Math.min(1, fraction) * 100}%` }} />
    </div>
  );
}
