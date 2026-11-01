/**
 * Just enough NIfTI reading for the browser: the header (to check shape and spacing before
 * upload) and one axial slice (for the preview while the server segments the study).
 * Handles NIfTI-1 and NIfTI-2, both byte orders, gzipped or not. Gzip is detected from the
 * content and decompressed as a stream, so a large volume is never held in memory whole.
 */

export type NiftiHeader = {
  version: 1 | 2;
  /** Size along each dimension (dim[1..dim[0]]). */
  shape: number[];
  /** Voxel size in mm along the first three axes. */
  spacing: number[];
  datatype: number;
  voxOffset: number;
  littleEndian: boolean;
  gzip: boolean;
  sclSlope: number;
  sclInter: number;
};

const BYTES_PER_VOXEL: Record<number, number> = {
  2: 1, 4: 2, 8: 4, 16: 4, 64: 8, 256: 1, 512: 2, 768: 4,
};

export class NiftiError extends Error {}

async function isGzip(file: Blob) {
  const head = new Uint8Array(await file.slice(0, 2).arrayBuffer());
  return head[0] === 0x1f && head[1] === 0x8b;
}

/** Decompressed (if needed) bytes in [start, end), reading the stream only as far as needed. */
async function readRange(file: Blob, gzip: boolean, start: number, end: number): Promise<Uint8Array> {
  if (!gzip) return new Uint8Array(await file.slice(start, end).arrayBuffer());
  const out = new Uint8Array(end - start);
  const reader = file.stream().pipeThrough(new DecompressionStream("gzip")).getReader();
  let pos = 0;
  try {
    while (pos < end) {
      const { done, value } = await reader.read();
      if (done) break;
      const from = Math.max(start - pos, 0);
      const to = Math.min(end - pos, value.length);
      if (to > from) out.set(value.subarray(from, to), pos + from - start);
      pos += value.length;
    }
  } finally {
    reader.cancel().catch(() => undefined);
  }
  if (pos < end) throw new NiftiError("the file ends before the image data does");
  return out;
}

export async function readHeader(file: Blob): Promise<NiftiHeader> {
  const gzip = await isGzip(file);
  let bytes: Uint8Array;
  try {
    bytes = await readRange(file, gzip, 0, 540);
  } catch (error) {
    if (error instanceof NiftiError) {
      // A NIfTI-1 header is 348 bytes; a tiny volume may end before 540.
      bytes = await readRange(file, gzip, 0, 348).catch(() => {
        throw new NiftiError("the file is too short to hold a NIfTI header");
      });
    } else {
      throw new NiftiError(gzip ? "the gzip data could not be decompressed" : "the file could not be read");
    }
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (const littleEndian of [true, false]) {
    const size = view.getInt32(0, littleEndian);
    if (size === 348) {
      const ndim = view.getInt16(40, littleEndian);
      if (ndim < 1 || ndim > 7) break;
      const dims = Array.from({ length: ndim }, (_, i) => view.getInt16(42 + 2 * i, littleEndian));
      return {
        version: 1,
        shape: dims,
        spacing: [1, 2, 3].map((i) => view.getFloat32(76 + 4 * i, littleEndian)),
        datatype: view.getInt16(70, littleEndian),
        voxOffset: Math.max(view.getFloat32(108, littleEndian), 352),
        littleEndian,
        gzip,
        sclSlope: view.getFloat32(112, littleEndian),
        sclInter: view.getFloat32(116, littleEndian),
      };
    }
    if (size === 540 && bytes.length >= 540) {
      const ndim = Number(view.getBigInt64(16, littleEndian));
      if (ndim < 1 || ndim > 7) break;
      const dims = Array.from({ length: ndim }, (_, i) => Number(view.getBigInt64(24 + 8 * i, littleEndian)));
      return {
        version: 2,
        shape: dims,
        spacing: [1, 2, 3].map((i) => view.getFloat64(104 + 8 * i, littleEndian)),
        datatype: view.getInt16(12, littleEndian),
        voxOffset: Number(view.getBigInt64(168, littleEndian)),
        littleEndian,
        gzip,
        sclSlope: view.getFloat64(176, littleEndian),
        sclInter: view.getFloat64(184, littleEndian),
      };
    }
  }
  throw new NiftiError("no NIfTI header found");
}

/** One axial slice (index z) as a Float32Array of shape[0] x shape[1], x varying fastest. */
async function readAxial(file: Blob, h: NiftiHeader, z: number): Promise<Float32Array> {
  const [nx, ny] = h.shape;
  const size = BYTES_PER_VOXEL[h.datatype];
  if (!size) throw new NiftiError(`unsupported data type ${h.datatype}`);
  const start = h.voxOffset + z * nx * ny * size;
  const raw = await readRange(file, h.gzip, start, start + nx * ny * size);
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const le = h.littleEndian;
  const read: Record<number, (o: number) => number> = {
    2: (o) => view.getUint8(o),
    4: (o) => view.getInt16(o, le),
    8: (o) => view.getInt32(o, le),
    16: (o) => view.getFloat32(o, le),
    64: (o) => view.getFloat64(o, le),
    256: (o) => view.getInt8(o),
    512: (o) => view.getUint16(o, le),
    768: (o) => view.getUint32(o, le),
  };
  const get = read[h.datatype];
  const slope = h.sclSlope && Number.isFinite(h.sclSlope) ? h.sclSlope : 1;
  const inter = Number.isFinite(h.sclInter) ? h.sclInter : 0;
  const out = new Float32Array(nx * ny);
  for (let i = 0; i < out.length; i++) out[i] = get(i * size) * slope + inter;
  return out;
}

/**
 * The axial slice rendered like the server's viewer layers: display orientation, cropped to the
 * brain with a small margin, windowed to the 0.5-99.5 percentile of the brain, transparent outside.
 * Returns a PNG data URL, or null if the slice is empty.
 */
export async function axialPreview(file: Blob, h: NiftiHeader, z = Math.floor(h.shape[2] / 2)) {
  const [nx, ny] = h.shape;
  const data = await readAxial(file, h, z);
  // Display orientation: row r shows y = ny - 1 - r, column c shows x = c.
  let r0 = ny, r1 = -1, c0 = nx, c1 = -1;
  const values: number[] = [];
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const v = data[x + y * nx];
      if (v === 0) continue;
      const r = ny - 1 - y;
      r0 = Math.min(r0, r); r1 = Math.max(r1, r); c0 = Math.min(c0, x); c1 = Math.max(c1, x);
      values.push(v);
    }
  }
  if (!values.length) return null;
  const pad = 6;
  r0 = Math.max(r0 - pad, 0); r1 = Math.min(r1 + pad, ny - 1);
  c0 = Math.max(c0 - pad, 0); c1 = Math.min(c1 + pad, nx - 1);
  values.sort((a, b) => a - b);
  const lo = values[Math.floor(values.length * 0.005)];
  const hi = Math.max(values[Math.floor(values.length * 0.995)], lo + 1e-6);

  const w = c1 - c0 + 1, hgt = r1 - r0 + 1;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = hgt;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const img = ctx.createImageData(w, hgt);
  for (let r = 0; r < hgt; r++) {
    const y = ny - 1 - (r + r0);
    for (let c = 0; c < w; c++) {
      const v = data[c + c0 + y * nx];
      const g = Math.round(255 * Math.pow(Math.min(Math.max((v - lo) / (hi - lo), 0), 1), 0.9));
      const o = 4 * (r * w + c);
      img.data[o] = img.data[o + 1] = img.data[o + 2] = g;
      img.data[o + 3] = v === 0 ? 0 : 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return { url: canvas.toDataURL("image/png"), index: z, width: w, height: hgt };
}
