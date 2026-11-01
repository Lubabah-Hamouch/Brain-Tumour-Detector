"use client";

import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { GLOSSARY, type GlossaryKey } from "@/lib/glossary";

import styles from "./Term.module.css";

type Position = { left: number; top: number; above: boolean };

const WIDTH = 288;
const GAP = 8;

/**
 * A domain term with a small "?" that reveals a plain-language definition on hover, focus or tap.
 * The definition is drawn above the page (a portal), so tables and scroll boxes never clip it.
 */
export function Term({ k, children }: { k: GlossaryKey; children?: ReactNode }) {
  const entry = GLOSSARY[k];
  const id = useId();
  const button = useRef<HTMLButtonElement>(null);
  const [pos, setPos] = useState<Position | null>(null);
  const [pinned, setPinned] = useState(false);

  const show = useCallback(() => {
    const r = button.current?.getBoundingClientRect();
    if (!r) return;
    const vw = document.documentElement.clientWidth;
    const left = Math.min(Math.max(16, r.left + r.width / 2 - WIDTH / 2), vw - WIDTH - 16);
    const above = r.bottom + GAP + 160 > window.innerHeight && r.top > 200;
    setPos({ left: Math.max(16, left), top: above ? r.top - GAP : r.bottom + GAP, above });
  }, []);
  const hide = useCallback(() => {
    setPos(null);
    setPinned(false);
  }, []);

  useEffect(() => {
    if (!pos) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && hide();
    const onPointer = (e: PointerEvent) => {
      if (!button.current?.contains(e.target as Node)) hide();
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onPointer);
    window.addEventListener("scroll", hide, true);
    window.addEventListener("resize", hide);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onPointer);
      window.removeEventListener("scroll", hide, true);
      window.removeEventListener("resize", hide);
    };
  }, [pos, hide]);

  return (
    <span className={styles.term}>
      {children}
      <button
        ref={button}
        type="button"
        className={styles.mark}
        aria-label={`What is ${entry.term}?`}
        aria-expanded={Boolean(pos)}
        aria-describedby={pos ? id : undefined}
        onMouseEnter={show}
        onMouseLeave={() => !pinned && setPos(null)}
        onFocus={show}
        onBlur={() => !pinned && setPos(null)}
        onClick={() => {
          if (pinned) hide();
          else {
            show();
            setPinned(true);
          }
        }}
      >
        ?
      </button>
      {pos &&
        createPortal(
          <span
            id={id}
            role="tooltip"
            className={styles.popover}
            data-above={pos.above}
            style={{ left: pos.left, top: pos.top, width: WIDTH }}
          >
            <strong>{entry.term}</strong>
            <span>{entry.text}</span>
          </span>,
          document.body,
        )}
    </span>
  );
}
