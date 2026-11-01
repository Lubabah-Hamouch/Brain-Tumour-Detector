"use client";

import { useEffect, useState } from "react";

import { easeOutCubic, motion } from "./tokens";

/**
 * Eased 0 -> 1 progress that runs once when the component mounts, for count-up
 * numbers and the bars that fill with them. Jumps straight to 1 when the viewer
 * prefers reduced motion.
 */
export function useEntryProgress(
  duration: number = motion.durationMs.countUp,
  delay: number = motion.durationMs.countUpDelay,
): number {
  const [t, setT] = useState(0);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let frame = 0;
    if (reduce) {
      frame = requestAnimationFrame(() => setT(1));
      return () => cancelAnimationFrame(frame);
    }
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, Math.max(0, (now - start - delay) / duration));
      setT(p);
      if (p < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [duration, delay]);

  return easeOutCubic(t);
}
