import type { Sample } from "@/lib/api";
import type { ImageInfo } from "@/lib/imageCheck";

/** A file that passed the browser checks and is ready to classify. */
export type Selected = { file: File; info: ImageInfo; previewUrl: string; sample?: Sample };

/** A file that was rejected, with the reason shown to the visitor. */
export type Problem = { name: string; bytes: number; message: string };
