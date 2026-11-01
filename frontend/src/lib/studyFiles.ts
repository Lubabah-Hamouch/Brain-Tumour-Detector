/**
 * The four-sequence upload for segmentation: matching files to sequences and checking them
 * before upload. The server checks again (backend/services/segmenter.py); these checks give
 * an immediate, specific answer per file.
 */
import { NiftiError, readHeader, type NiftiHeader } from "./nifti";

export const SEQUENCES = ["FLAIR", "T1", "T1ce", "T2"] as const;
export type Sequence = (typeof SEQUENCES)[number];
/** API field name per sequence. */
export const FIELD: Record<Sequence, "flair" | "t1" | "t1ce" | "t2"> = {
  FLAIR: "flair", T1: "t1", T1ce: "t1ce", T2: "t2",
};

export const MAX_VOLUME_MB = 200;
export const EXPECTED_SPACING_MM = 1;

/**
 * Sequence from a file name. Contrast-enhanced T1 is tested before plain T1, since its names
 * (t1ce, t1c, t1gd, t1_ce, t1-ce) all contain "t1".
 */
export function detectSequence(name: string): Sequence | null {
  const n = name.toLowerCase().replace(/\.nii(\.gz)?$/, "");
  if (/t1[_-]?ce|t1c(?![a-z])|t1gd|t1[_-]?gd|t1[_-]?post|ce[_-]?t1/.test(n)) return "T1ce";
  if (/flair|t2[_-]?f(?![a-z])/.test(n)) return "FLAIR";
  if (/t2/.test(n)) return "T2";
  if (/t1/.test(n)) return "T1";
  return null;
}

/** What the browser could read from one file, independent of the other files. */
export type FileCheck =
  | { kind: "pending" }
  | { kind: "nifti"; header: NiftiHeader }
  | { kind: "rejected"; message: string };

export type StudyFile = {
  id: string;
  file: File;
  detected: Sequence | null;
  /** Set when the visitor overrides the detected sequence. */
  assigned: Sequence | null;
  check: FileCheck;
};

export const sequenceOf = (f: StudyFile) => f.assigned ?? f.detected;

export async function checkFile(file: File): Promise<FileCheck> {
  const name = file.name.toLowerCase();
  if (name.endsWith(".dcm") || name.endsWith(".dicom")) {
    return { kind: "rejected", message: "DICOM is not supported; convert the series to NIfTI first" };
  }
  if (!name.endsWith(".nii") && !name.endsWith(".nii.gz")) {
    return { kind: "rejected", message: "Not a NIfTI file (.nii or .nii.gz); it will not be uploaded" };
  }
  if (file.size === 0) return { kind: "rejected", message: "The file is empty" };
  if (file.size > MAX_VOLUME_MB * 1024 * 1024) {
    return { kind: "rejected", message: `${(file.size / 1024 / 1024).toFixed(0)} MB is over the ${MAX_VOLUME_MB} MB limit` };
  }
  try {
    const header = await readHeader(file);
    const shape = header.shape.length === 4 && header.shape[3] === 1 ? header.shape.slice(0, 3) : header.shape;
    if (shape.length !== 3) {
      return { kind: "rejected", message: `Shape ${header.shape.join(" × ")}; each sequence must be one 3D volume` };
    }
    return { kind: "nifti", header: { ...header, shape } };
  } catch (error) {
    const reason = error instanceof NiftiError ? error.message : "the file could not be read";
    return { kind: "rejected", message: `Not a readable NIfTI file: ${reason}` };
  }
}

const sameGrid = (a: NiftiHeader, b: NiftiHeader) =>
  a.shape.every((s, i) => s === b.shape[i]) && a.spacing.every((s, i) => Math.abs(s - b.spacing[i]) < 1e-3);

export const formatShape = (h: NiftiHeader) => h.shape.join(" × ");
export const formatSpacing = (h: NiftiHeader) =>
  h.spacing.every((s) => Math.abs(s - h.spacing[0]) < 1e-3)
    ? `${h.spacing[0].toFixed(1)} mm`
    : `${h.spacing.map((s) => s.toFixed(2)).join(" × ")} mm`;

export type RowStatus = { ok: boolean; pending?: boolean; message: string; source: string };

/** Status of every file in the context of the others: sequence conflicts and grid mismatches. */
export function evaluate(files: StudyFile[]) {
  const counts = new Map<Sequence, number>();
  for (const f of files) {
    const seq = sequenceOf(f);
    if (f.check.kind === "nifti" && seq) counts.set(seq, (counts.get(seq) ?? 0) + 1);
  }
  // The grid every file must share: FLAIR's if it is readable, else the first readable file's.
  const readable = files.filter((f) => f.check.kind === "nifti");
  const reference = readable.find((f) => sequenceOf(f) === "FLAIR") ?? readable[0];
  const refHeader = reference?.check.kind === "nifti" ? reference.check.header : null;

  const status = new Map<string, RowStatus>();
  for (const f of files) {
    const seq = sequenceOf(f);
    const source =
      f.check.kind === "rejected" ? "Unsupported" : f.assigned ? "Set manually" : seq ? "Detected from file name" : "Not detected";
    if (f.check.kind === "pending") {
      status.set(f.id, { ok: false, pending: true, message: "Reading the header…", source });
    } else if (f.check.kind === "rejected") {
      status.set(f.id, { ok: false, message: f.check.message, source });
    } else if (!seq) {
      status.set(f.id, { ok: false, message: "Sequence not recognised from the name; choose one", source });
    } else if ((counts.get(seq) ?? 0) > 1) {
      status.set(f.id, { ok: false, message: `Two files are set to ${seq}; reassign one`, source });
    } else if (refHeader && f !== reference && !sameGrid(f.check.header, refHeader)) {
      const ref = `${formatShape(refHeader)} · ${formatSpacing(refHeader)}`;
      status.set(f.id, {
        ok: false,
        message: `Grid ${formatShape(f.check.header)} · ${formatSpacing(f.check.header)} differs from ${sequenceOf(reference!) ?? "the other files"} (${ref})`,
        source,
      });
    } else {
      const h = f.check.header;
      const off = h.spacing.some((s) => Math.abs(s - EXPECTED_SPACING_MM) > 0.1);
      status.set(f.id, {
        ok: true,
        message: `NIfTI · ${formatShape(h)} · ${formatSpacing(h)}${off ? " · not 1 mm, runs with a warning" : ""}`,
        source,
      });
    }
  }
  const slots = SEQUENCES.map((name) => ({
    name,
    file: files.find((f) => sequenceOf(f) === name && status.get(f.id)?.ok) ?? null,
  }));
  return { status, slots, ready: slots.every((s) => s.file) };
}
