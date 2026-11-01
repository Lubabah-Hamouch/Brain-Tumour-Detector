"use client";

import { useState } from "react";

import { chartSeries } from "@/design/tokens";
import { useEntryProgress } from "@/design/useEntryProgress";
import type { ModelCard } from "@/lib/api";

import styles from "./ModelCard.module.css";

type Data = ModelCard["stress_test"];

/** Smallest "nice" axis maximum (1, 2, 2.5 or 5 times a power of ten) at or above `value`. */
function niceMax(value: number) {
  if (value <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  return [1, 2, 2.5, 5, 10].map((m) => m * power).find((m) => m >= value)!;
}

const seriesLabel = (m: Data["models"][number]) => `${m.key} · ${m.served ? "Served" : m.name}`;

/**
 * Grouped bars: per size condition, how many tumour images each classifier variant calls
 * "no tumour". One axis, values labelled on every bar (16 small counts, the point of the chart),
 * legend above, tooltip on hover or focus with the count and overall accuracy.
 */
export function StressChart({ data }: { data: Data }) {
  const e = useEntryProgress();
  const [active, setActive] = useState<{ g: number; s: number } | null>(null);
  const max = niceMax(Math.max(...data.models.flatMap((m) => m.values)));
  const ticks = [max, max / 2, 0];
  const plotHeight = 280;
  const labelRoom = 32; // space above the tallest bar for its value
  const y = (v: number) => (v / max) * (plotHeight - labelRoom);

  const description = `Grouped bar chart. Tumour images out of ${data.tumour_images} called no tumour. ${data.conditions
    .map((c, g) => `${c.label}: ${data.models.map((m) => `${m.key} ${m.values[g]}`).join(", ")}`)
    .join(". ")}.`;

  return (
    <div className={styles.chartBlock}>
      <div className={styles.legend}>
        {data.models.map((m, s) => (
          <span key={m.key} className={styles.legendItem} data-served={m.served}>
            <span className={styles.swatch} style={{ background: chartSeries[s] }} />
            {seriesLabel(m)}
          </span>
        ))}
      </div>
      <div className={styles.chartScroll}>
        <div className={styles.chart} role="group" aria-label={description} onMouseLeave={() => setActive(null)}>
          <div className={`mono ${styles.yAxis}`} style={{ height: plotHeight }} aria-hidden="true">
            {ticks.map((t) => (
              <span key={t} style={{ bottom: y(t) }}>
                {t}
              </span>
            ))}
          </div>
          {data.conditions.map((c, g) => (
            <div key={c.key} className={styles.group} style={{ height: plotHeight }}>
              {data.models.map((m, s) => {
                const value = m.values[g];
                const isActive = active?.g === g && active?.s === s;
                return (
                  <div
                    key={m.key}
                    className={styles.barSlot}
                    data-dim={active !== null && !isActive}
                    tabIndex={0}
                    aria-label={`${seriesLabel(m)}, ${c.label}: ${value} of ${data.tumour_images}`}
                    onMouseEnter={() => setActive({ g, s })}
                    onFocus={() => setActive({ g, s })}
                    onBlur={() => setActive(null)}
                  >
                    <span className={`mono ${styles.barValue}`} data-served={m.served}>
                      {Math.round(value * e)}
                    </span>
                    <div
                      className={styles.bar}
                      style={{
                        height: Math.max(1, y(value) * e),
                        background: chartSeries[s],
                      }}
                    />
                    {isActive && (
                      <span
                        className={styles.tooltip}
                        role="tooltip"
                        style={{ bottom: Math.max(1, y(value)) + 28 }}
                        data-edge={g === 0 ? "start" : g === data.conditions.length - 1 ? "end" : undefined}
                      >
                        <strong>{seriesLabel(m)}</strong>
                        <span>{c.label}</span>
                        <span className="mono">
                          {value} of {data.tumour_images} tumours called no tumour
                        </span>
                        <span className="mono">accuracy {(m.accuracy[g] * 100).toFixed(2)}%</span>
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
          <div aria-hidden="true" />
          {data.conditions.map((c) => (
            <div key={c.key} className={`mono ${styles.groupLabel}`} aria-hidden="true">
              {c.label}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
