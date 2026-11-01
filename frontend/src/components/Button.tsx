import type { ComponentProps } from "react";

import styles from "./ButtonLink.module.css";

type Props = ComponentProps<"button"> & { variant?: "solid" | "ghost"; block?: boolean };

/** An action button with the same look as ButtonLink. */
export function Button({ variant = "solid", block = false, className, type = "button", ...props }: Props) {
  const classes = [styles.button, styles[variant], block ? styles.block : "", className ?? ""];
  return <button type={type} {...props} className={classes.join(" ")} />;
}
