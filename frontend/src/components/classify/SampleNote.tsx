import { CLASS_LABELS, type Sample } from "@/lib/api";

/** "meningioma", or "glioma · known error" for a sample the model gets wrong on the test set. */
export function SampleNote({ sample, suffix }: { sample: Sample; suffix?: string }) {
  const truth = (CLASS_LABELS[sample.true_class] ?? sample.true_class).toLowerCase();
  const parts = [truth, sample.known_error ? "known error" : "", suffix ?? ""].filter(Boolean);
  return <span className="caption">{parts.join(" · ")}</span>;
}
