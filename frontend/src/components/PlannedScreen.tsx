import { ButtonLink } from "./ButtonLink";
import styles from "./PlannedScreen.module.css";

/** Placeholder for a screen that exists in the design but is not built yet. */
export function PlannedScreen({ label, title, body }: { label: string; title: string; body: string }) {
  return (
    <main className={styles.main}>
      <p className="label">{label}</p>
      <h1 className={styles.title}>{title}</h1>
      <p className={styles.body}>{body}</p>
      <ButtonLink href="/" variant="ghost">
        Back to overview
      </ButtonLink>
    </main>
  );
}
