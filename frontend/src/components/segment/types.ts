import type { SequenceField } from "@/lib/api";
import type { NiftiHeader } from "@/lib/nifti";

/** The local FLAIR's axial slice, shown dimmed while the server works. */
export type Preview = { url: string; index: number; width: number; height: number };

/** One press of "Run segmentation". */
export type Run = {
  id: number;
  /** performance.now() when the visitor pressed Run. */
  startedAt: number;
  header: NiftiHeader | null;
  /** The uploaded files; null for the sample study's pre-computed result, which uploads nothing. */
  files: Record<SequenceField, File> | null;
  /** The sample case's ID when the run is the sample study. */
  sample: string | null;
  /** Set once the upload is accepted and the job exists. */
  statusUrl: string | null;
  /** Upload plus the server's synchronous checks (or the sample request), measured in the browser. */
  uploadSeconds: number | null;
  preview: Preview | null;
};
