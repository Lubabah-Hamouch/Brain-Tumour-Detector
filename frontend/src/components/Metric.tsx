"use client";

import { useEntryProgress } from "@/design/useEntryProgress";

import styles from "./Metric.module.css";

type Format = { decimals: number; scale?: number };

const format = (value: number, { decimals, scale = 1 }: Format) => (value * scale).toFixed(decimals);

/** A large number that counts up from zero on entry, with a bar that fills in step. */
export function HeroMetric({
  label,
  value,
  decimals,
  scale,
  unit,
  barMax = 1,
}: {
  label: string;
  value: number;
  decimals: number;
  scale?: number;
  unit?: string;
  barMax?: number;
}) {
  const e = useEntryProgress();
  return (
    <div className={styles.hero}>
      <span className="label">{label}</span>
      <span className={`mono ${styles.heroValue}`} aria-label={`${format(value, { decimals, scale })}${unit ?? ""}`}>
        <span aria-hidden="true">
          {format(value * e, { decimals, scale })}
          {unit && <span className={styles.unit}>{unit}</span>}
        </span>
      </span>
      <Bar fraction={(value / barMax) * e} strong />
    </div>
  );
}

/** A compact row: name, thin bar and value, counting up with the hero number above it. */
export function MetricRow({ name, value, decimals }: { name: string; value: number; decimals: number }) {
  const e = useEntryProgress();
  return (
    <div className={styles.row}>
      <span>{name}</span>
      <Bar fraction={value * e} />
      <span className={styles.rowValue}>{(value * e).toFixed(decimals)}</span>
    </div>
  );
}

export function Bar({ fraction, strong = false }: { fraction: number; strong?: boolean }) {
  return (
    <div aria-hidden="true" className={strong ? styles.barStrong : styles.bar}>
      <div className={styles.fill} style={{ width: `${Math.max(0, Math.min(1, fraction)) * 100}%` }} />
    </div>
  );
}
