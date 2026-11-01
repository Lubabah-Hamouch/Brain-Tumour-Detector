import Link from "next/link";
import type { ComponentProps } from "react";

import styles from "./ButtonLink.module.css";

type Props = ComponentProps<typeof Link> & { variant?: "solid" | "ghost" };

/** Navigation styled as a button: solid off-white for the primary action, hairline ghost for the rest. */
export function ButtonLink({ variant = "solid", className, ...props }: Props) {
  return <Link {...props} className={`${styles.button} ${styles[variant]} ${className ?? ""}`} />;
}
