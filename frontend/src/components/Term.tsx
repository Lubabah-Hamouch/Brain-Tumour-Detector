"use client";

import { useCallback, useEffect, useId, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { GLOSSARY, type GlossaryKey } from "@/lib/glossary";

import styles from "./Term.module.css";

type Position = { left: number; top: number; above: boolean };

const WIDTH = 288;
const GAP = 8;

/**
 * A glossary definition anchored to an element: position, dismissal (Escape, outside press, scroll,
 * resize) and the popover itself, drawn above the page (a portal) so tables and scroll boxes never
 * clip it.
 */
export function useDefinition<T extends HTMLElement>(k: GlossaryKey) {
  const entry = GLOSSARY[k];
  const id = useId();
  // A callback ref kept in state, so the element is a normal value during render.
  const [anchor, setAnchor] = useState<T | null>(null);
  const [pos, setPos] = useState<Position | null>(null);
  const [pinned, setPinned] = useState(false);

  const show = useCallback(() => {
    const r = anchor?.getBoundingClientRect();
    if (!r) return;
    const vw = document.documentElement.clientWidth;
    const left = Math.min(Math.max(16, r.left + r.width / 2 - WIDTH / 2), vw - WIDTH - 16);
    const above = r.bottom + GAP + 160 > window.innerHeight && r.top > 200;
    setPos({ left: Math.max(16, left), top: above ? r.top - GAP : r.bottom + GAP, above });
  }, [anchor]);
  const hide = useCallback(() => {
    setPos(null);
    setPinned(false);
  }, []);

  useEffect(() => {
    if (!pos) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && hide();
    const onPointer = (e: PointerEvent) => {
      if (!anchor?.contains(e.target as Node)) hide();
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
  }, [pos, hide, anchor]);

  const popover =
    pos &&
    createPortal(
      <span id={id} role="tooltip" className={styles.popover} data-above={pos.above} style={{ left: pos.left, top: pos.top, width: WIDTH }}>
        <strong>{entry.term}</strong>
        <span>{entry.text}</span>
      </span>,
      document.body,
    );

  return { entry, id, setAnchor, open: Boolean(pos), pinned, setPinned, show, hide, popover };
}

/**
 * A domain term with a small "?" that reveals a plain-language definition on hover, focus or tap.
 */
export function Term({ k, children }: { k: GlossaryKey; children?: ReactNode }) {
  const { entry, id, setAnchor, open, pinned, setPinned, show, hide, popover } = useDefinition<HTMLButtonElement>(k);

  return (
    <span className={styles.term}>
      {children}
      <button
        ref={setAnchor}
        type="button"
        className={styles.mark}
        aria-label={`What is ${entry.term}?`}
        aria-expanded={open}
        aria-describedby={open ? id : undefined}
        onMouseEnter={show}
        onMouseLeave={() => !pinned && hide()}
        onFocus={show}
        onBlur={() => !pinned && hide()}
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
      {popover}
    </span>
  );
}
