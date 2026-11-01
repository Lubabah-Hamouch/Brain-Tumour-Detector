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
  files: Record<SequenceField, File>;
  /** The sample case's ID when the files came from the sample study. */
  sample: string | null;
  /** Set once the upload is accepted and the job exists. */
  statusUrl: string | null;
  /** Upload plus the server's synchronous checks, measured in the browser. */
  uploadSeconds: number | null;
  preview: Preview | null;
};
