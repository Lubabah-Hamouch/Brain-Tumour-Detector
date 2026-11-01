/** Status marks used next to file and stage states. They take the text colour of their row. */

export function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
      <path d="M2.5 7.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

export function ErrorIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
      <circle cx="7" cy="7" r="6" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M7 3.8v4M7 9.6v0.6" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}
