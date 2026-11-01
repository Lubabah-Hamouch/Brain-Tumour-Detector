import { existsSync } from "node:fs";
import path from "node:path";

import Image from "next/image";
import Link from "next/link";

import { regionColor } from "@/design/tokens";

import styles from "./HeroVisual.module.css";

// The hero is rendered from patient data, so it is generated locally and not committed
// (see src/visualization/render_hero.py). Checked once, when the server starts.
const HERO_SRC = "/generated/hero-3d.png";
const HERO_AVAILABLE = existsSync(path.join(process.cwd(), "public", "generated", "hero-3d.png"));

const LEGEND = [
  { label: "Whole tumour (edema)", color: regionColor.WT },
  { label: "Necrotic core", color: regionColor.TC },
  { label: "Enhancing", color: regionColor.ET },
];

/**
 * Static 3D render of a held-out test patient. It links to the live version: the Segment screen
 * runs the same patient through the models and opens the result in its interactive 3D view.
 */
export function HeroVisual() {
  return (
    <figure className={styles.figure}>
      <div className={styles.frame}>
        {HERO_AVAILABLE ? (
          <Link href="/segment?demo=3d" className={styles.link} aria-label="View this patient in 3D: run the segmentation and open the interactive 3D view">
            <Image
              src={HERO_SRC}
              alt="3D render of held-out test patient BraTS2021_00003: a translucent brain surface with the ensemble's predicted tumour regions inside it in shades of red"
              width={996}
              height={1200}
              priority
              className={styles.image}
            />
            <div aria-hidden="true" className={styles.sweep} />
            <span className={styles.cta} aria-hidden="true">
              <span className="u">View in 3D</span> →
            </span>
          </Link>
        ) : (
          <p className={styles.missing}>
            Hero image not generated yet. From the project root run{" "}
            <code className="mono">python -m src.visualization.render_hero</code>.
          </p>
        )}
      </div>
      <figcaption className={styles.caption}>
        <span className={styles.legend}>
          {LEGEND.map((item) => (
            <span key={item.label}>
              <span className={styles.swatch} style={{ background: item.color }} />
              {item.label}
            </span>
          ))}
        </span>
        <span>
          BraTS2021_00003 · held-out test patient · 3D surfaces of the brain and the ensemble prediction ·{" "}
          <Link href="/segment?demo=3d" className={styles.captionLink}>
            <span className="u">open it live in 3D</span>
          </Link>
        </span>
      </figcaption>
    </figure>
  );
}
