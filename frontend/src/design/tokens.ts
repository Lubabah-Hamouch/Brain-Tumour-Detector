/**
 * Design tokens: the single source for colour, type, motion and spacing.
 *
 * `cssVariables()` turns them into CSS custom properties on :root (injected by the
 * root layout), so stylesheets use `var(--…)` and TypeScript uses these constants;
 * no value is written twice.
 *
 * Colour rule: the interface is greyscale. The red family appears only where the
 * models flag something (tumour regions, Grad-CAM, the detected class).
 */

export const color = {
  // Surfaces and lines
  bg: "#131416",
  bgExport: "#0B0C0D",
  hairline: "#24262A",
  hairlineStrong: "#3A3D42",
  control: "#2E3034",
  barTrack: "#2A2C30",
  barFill: "#E4E6E8",
  barMuted: "#9EA2A7",
  // Text, from loudest to quietest
  textStrong: "#FFFFFF",
  text: "#ECEDEF",
  textSecondary: "#9A9EA4",
  textMuted: "#858A90",
  textDisabled: "#6E7379",
  // Buttons
  solid: "#F2F2F0",
  solidText: "#0B0C0D",
  // Findings only: BraTS labels 2 (edema), 1 (necrotic core), 4 (enhancing tumour)
  edema: "#7A111B",
  necroticCore: "#D7263D",
  enhancing: "#FFC2CA",
} as const;

/** Region swatches used in legends and results (WT contains TC contains ET). */
export const regionColor = { WT: color.edema, TC: color.necroticCore, ET: color.enhancing } as const;

/** Fixed type scale in px; nothing in between. */
export const typeScale = [12, 14, 16, 20, 28, 40, 64, 128] as const;
export type TypeStep = (typeof typeScale)[number];

export const lineHeight: Record<TypeStep, number> = {
  128: 0.9, 64: 1, 40: 1, 28: 1.25, 20: 1.4, 16: 1.65, 14: 1.65, 12: 1.5,
};

/** Tracking per step: tight at display sizes, wide for 12 px uppercase labels. */
export const letterSpacing: Record<TypeStep, string> = {
  128: "-0.05em", 64: "-0.035em", 40: "-0.03em", 28: "-0.02em", 20: "-0.01em", 16: "0", 14: "0", 12: "0.16em",
};

export const motion = {
  /** The only easing curve: a quick ease-out with no overshoot. */
  ease: "cubic-bezier(0.2, 0.7, 0.2, 1)",
  durationMs: {
    hover: 180,
    underline: 240,
    crossfade: 320,
    countUp: 1400,
    countUpDelay: 250,
    resolve: 1600,
    sweep: 1800,
    sweepDelay: 700,
  },
} as const;

/** JS counterpart of the easing curve for values animated in script (count-ups). */
export const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);

export const space = {
  pageMargin: 64,
  sectionGap: 200,
  dotGrid: 24,
} as const;

const kebab = (s: string) => s.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);

export function cssVariables(): string {
  const vars: string[] = [];
  for (const [k, v] of Object.entries(color)) vars.push(`--color-${kebab(k)}:${v}`);
  for (const [k, v] of Object.entries(regionColor)) vars.push(`--region-${k.toLowerCase()}:${v}`);
  for (const step of typeScale) {
    vars.push(`--text-${step}:${step}px`, `--lh-${step}:${lineHeight[step]}`, `--track-${step}:${letterSpacing[step]}`);
  }
  vars.push(`--ease:${motion.ease}`);
  for (const [k, v] of Object.entries(motion.durationMs)) vars.push(`--dur-${kebab(k)}:${v}ms`);
  vars.push(`--page-margin:${space.pageMargin}px`, `--section-gap:${space.sectionGap}px`, `--dot-grid:${space.dotGrid}px`);
  return `:root{${vars.join(";")}}`;
}
