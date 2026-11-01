import styles from "./SiteFooter.module.css";

export function SiteFooter() {
  return (
    <footer className={styles.footer}>
      <p>Not a medical device. A research demonstration; outputs must not inform diagnosis or treatment.</p>
      <p>Data: BraTS 2021 · Brain Tumor MRI dataset (Figshare, SARTAJ, Br35H)</p>
    </footer>
  );
}
