/**
 * Browser-side checks for a classification upload, mirroring what the API enforces
 * (backend/routers/classification.py): JPEG or PNG, at most 20 MB, at least 32 px a side.
 * The server checks again; this only gives an immediate, specific answer.
 */

export const MAX_IMAGE_MB = 20;
export const MIN_SIDE = 32;

export type ImageInfo = {
  name: string;
  format: "JPEG" | "PNG";
  width: number;
  height: number;
  bytes: number;
  grayscale: boolean;
};

export type CheckResult = { ok: true; info: ImageInfo; previewUrl: string } | { ok: false; message: string };

const ALLOWED = [".jpg", ".jpeg", ".png"];

function extension(name: string) {
  const dot = name.lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot).toLowerCase();
}

/** Format from the file's first bytes, not its name. */
function sniff(head: Uint8Array): "JPEG" | "PNG" | "DICOM" | null {
  if (head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff) return "JPEG";
  if (head[0] === 0x89 && head[1] === 0x50 && head[2] === 0x4e && head[3] === 0x47) return "PNG";
  // DICOM Part 10: 128-byte preamble, then "DICM".
  if (head.length >= 132 && String.fromCharCode(...head.slice(128, 132)) === "DICM") return "DICOM";
  return null;
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** True when the red, green and blue channels agree, on a small downscaled copy. */
function isGrayscale(bitmap: ImageBitmap) {
  const side = 64;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = side;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return false;
  ctx.drawImage(bitmap, 0, 0, side, side);
  const px = ctx.getImageData(0, 0, side, side).data;
  for (let i = 0; i < px.length; i += 4) {
    if (Math.abs(px[i] - px[i + 1]) > 3 || Math.abs(px[i + 1] - px[i + 2]) > 3) return false;
  }
  return true;
}

export async function checkImage(file: File): Promise<CheckResult> {
  const ext = extension(file.name);
  const head = new Uint8Array(await file.slice(0, 132).arrayBuffer());
  const kind = sniff(head);

  if (kind === "DICOM" || ext === ".dcm" || ext === ".dicom") {
    return { ok: false, message: "DICOM is not supported; export the slice as JPEG or PNG" };
  }
  if (!ALLOWED.includes(ext)) {
    const what = ext ? `${ext.slice(1).toUpperCase()} files are` : "Files without an extension are";
    return { ok: false, message: `${what} not supported; use a JPEG or PNG` };
  }
  if (file.size === 0) return { ok: false, message: "The file is empty" };
  if (file.size > MAX_IMAGE_MB * 1024 * 1024) {
    return { ok: false, message: `The file is ${formatBytes(file.size)}; the limit is ${MAX_IMAGE_MB} MB` };
  }
  if (kind !== "JPEG" && kind !== "PNG") {
    return { ok: false, message: `The file is named ${ext} but is not a JPEG or PNG image` };
  }

  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    return { ok: false, message: `The ${kind} could not be decoded; it may be damaged` };
  }
  const { width, height } = bitmap;
  const grayscale = isGrayscale(bitmap);
  bitmap.close();
  if (Math.min(width, height) < MIN_SIDE) {
    return { ok: false, message: `The image is ${width} × ${height}; at least ${MIN_SIDE} px a side is needed` };
  }
  return {
    ok: true,
    info: { name: file.name, format: kind, width, height, bytes: file.size, grayscale },
    previewUrl: URL.createObjectURL(file),
  };
}
