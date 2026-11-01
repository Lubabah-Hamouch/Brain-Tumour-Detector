import { CLASS_LABELS, type Sample } from "@/lib/api";

/**
 * "meningioma", or "glioma · known error" for a sample the model gets wrong on the test set.
 * With `name`, the file name and the known-error flag instead of the class.
 */
export function SampleNote({ sample, suffix, name = false }: { sample: Sample; suffix?: string; name?: boolean }) {
  const truth = (CLASS_LABELS[sample.true_class] ?? sample.true_class).toLowerCase();
  const parts = [name ? sample.name : truth, sample.known_error ? "known error" : "", suffix ?? ""].filter(Boolean);
  return <span className="caption">{parts.join(" · ")}</span>;
}
