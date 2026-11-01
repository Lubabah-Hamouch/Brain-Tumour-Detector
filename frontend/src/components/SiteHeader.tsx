"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import styles from "./SiteHeader.module.css";

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/segment", label: "Segment" },
  { href: "/classify", label: "Classify" },
  { href: "/model-card", label: "Model card" },
] as const;

export function SiteHeader() {
  const pathname = usePathname();
  const isActive = (href: string) => (href === "/" ? pathname === "/" : pathname.startsWith(href));

  return (
    <header className={styles.header}>
      <Link href="/" className={styles.brand}>
        Brain Tumour MRI
      </Link>
      <nav aria-label="Main" className={styles.nav}>
        {NAV.map(({ href, label }) => {
          const active = isActive(href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              className={active ? styles.active : styles.link}
            >
              <span className="u">{label}</span>
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
