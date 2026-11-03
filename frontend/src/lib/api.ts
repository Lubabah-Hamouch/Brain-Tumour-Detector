/**
 * Typed access to the FastAPI backend (backend/). Server components call it with
 * API_URL; the browser uses NEXT_PUBLIC_API_URL, which is fixed into the client bundle at build
 * time. In development both default to the local server.
 *
 * A production build without an address is the static demo (STATIC_DEMO): the samples, the test
 * results and the model card come from files that deploy/static/export_samples.py saved from the
 * API into public/demo/, and uploads are not available, since nothing can run the models.
 */

const DEV_API_URL = "http://127.0.0.1:8010";
const configuredUrl = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL;

export const API_URL = (configuredUrl || (process.env.NODE_ENV === "production" ? "" : DEV_API_URL)).replace(/\/+$/, "");
export const STATIC_DEMO = !API_URL;
const DEMO = "/demo";

export type ResultsSummary = {
  segmentation: {
    test_patients: number;
    model: string;
    dice: { mean: number; WT: number; TC: number; ET: number };
  };
  classification: {
    test_images: number;
    model: string;
    checkpoint: string;
    accuracy: number;
    accuracy_ci95: [number, number];
    macro_f1: number;
    accuracy_at_128px: number | null;
    checkpoint_epoch: number;
    test_errors: number;
    /** Test errors per unordered class pair, most frequent first. */
    confusions: { classes: [string, string]; errors: number }[];
  };
  sources: string[];
};

export type ClassName = "glioma" | "meningioma" | "notumor" | "pituitary";

export const CLASS_LABELS: Record<string, string> = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  notumor: "No tumour",
  pituitary: "Pituitary",
};

export type ClassificationResult = {
  result_id: string;
  report_url: string;
  predicted_class: ClassName;
  confidence: number;
  probabilities: Record<ClassName, number>;
  image: { width: number; height: number; mode: string; grayscale: boolean };
  letterbox: { size: number; scale: number; offset_x: number; offset_y: number };
  /** Base64 PNG: the letterboxed grayscale slice the model saw. */
  model_input_png: string;
  gradcam: {
    target_class: string;
    /** Base64 PNG, RGBA: Grad-CAM on the red ramp, aligned with model_input_png. */
    heat_png: string;
    /**
     * Outline where the heatmap reaches `contour_level` of its peak: polylines of [x, y] in
     * model-input pixels; empty when there is nothing to outline. Optional because an API started
     * before the outline was added does not send it.
     */
    contour?: [number, number][][];
    contour_level?: number;
    resolution_note: string;
  };
  model: { checkpoint: string; checkpoint_epoch: number };
  warnings: string[];
  /** True for a sample image, answered from the result computed when the API started. */
  precomputed?: boolean;
};

export type Sample = {
  name: string;
  true_class: ClassName;
  size_bytes: number;
  known_error: boolean;
  recorded_prediction: string | null;
  url: string;
};

export class ApiUnavailable extends Error {}

/** An error the API answered with: `{"error": {"code", "message"}}`. */
export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!API_URL) {
    throw new ApiUnavailable(
      "This online demo has no server to run the models on, so it shows the samples only. To try your own " +
        "images, run the app on your computer (see the project's README).",
    );
  }
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch (cause) {
    throw new ApiUnavailable(`The API at ${API_URL} is not reachable.`, { cause });
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    if (body?.error) throw new ApiError(body.error.code, body.error.message);
    throw new ApiUnavailable(`The API returned ${response.status}.`);
  }
  return response.json();
}

/** A file of the static demo, as the API answered it (browser only). */
async function demoFile<T>(path: string): Promise<T> {
  const response = await fetch(`${DEMO}${path}`);
  if (!response.ok) throw new ApiUnavailable(`The demo file ${DEMO}${path} is missing (${response.status}).`);
  return response.json();
}

export async function getResultsSummary(): Promise<ResultsSummary> {
  // Imported, not fetched: server components read it too.
  if (STATIC_DEMO) return (await import("../../public/demo/summary.json")).default as unknown as ResultsSummary;
  return request("/api/results/summary");
}

export function getSamples(): Promise<Sample[]> {
  return STATIC_DEMO ? demoFile("/classify/samples.json") : request("/api/samples");
}

/** `sample`: the file is that sample, unchanged; the static demo answers it with the saved result. */
export function classify(file: File, sample?: Sample): Promise<ClassificationResult> {
  if (STATIC_DEMO && sample) return demoFile(`/classify/results/${sample.name.replace(/\.[^.]+$/, "")}.json`);
  const form = new FormData();
  form.append("file", file);
  return request("/api/classify", { method: "POST", body: form });
}

// ---- Segmentation ----

export type Plane = "axial" | "coronal" | "sagittal";
export type SequenceField = "flair" | "t1" | "t1ce" | "t2";
export type LabelLayer = "edema" | "necrotic" | "enhancing";

export type RegionResult = {
  voxels: number;
  volume_ml: number;
  mean_probability: number | null;
  model_agreement: number;
};

export type ViewerPlane = {
  index: number;
  count: number;
  width: number;
  height: number;
  layers: Record<SequenceField | LabelLayer, string>;
};

export type SegmentationResult = {
  result_id: string;
  study: string;
  files: Partial<Record<SequenceField, string>>;
  mask_url: string;
  report_url: string;
  /** Brain and tumour surfaces as binary glTF, built on first request. */
  mesh_url: string;
  regions: Record<"WT" | "TC" | "ET", RegionResult>;
  shape: number[];
  spacing_mm: number[];
  slices: Record<Plane, ViewerPlane>;
  post_processing: {
    et_min_component_voxels: number;
    et_components_predicted: number;
    et_components_removed: number;
    et_voxels_removed: number;
  };
  model: { folds: number[]; cross_validation_mean_dice: Record<string, number> };
  timing_seconds: { load: number; preprocess: number; inference: number; postprocess_and_encode: number };
  warnings: string[];
  /** True for the sample study: computed when the API started; `timing_seconds` are from that run. */
  precomputed?: boolean;
};

export type JobStage = {
  name: string;
  label: string;
  status: "pending" | "running" | "complete" | "failed";
  fraction: number;
  detail: string;
  seconds: number | null;
};

export type JobStatus = {
  job_id: string;
  status: "queued" | "running" | "complete" | "failed";
  queue_position: number | null;
  stage: string | null;
  stage_label: string | null;
  stage_detail: string | null;
  progress: number;
  stages: JobStage[];
  elapsed_seconds: number;
  result: SegmentationResult | null;
  error: { code: string; message: string } | null;
};

export type JobAccepted = { job_id: string; status: string; status_url: string };

export type SampleStudy = {
  case: string;
  split: string;
  files: { sequence: SequenceField; name: string; size_bytes: number; url: string }[];
};

export function startSegmentation(files: Record<SequenceField, File>): Promise<JobAccepted> {
  const form = new FormData();
  for (const [field, file] of Object.entries(files)) form.append(field, file);
  return request("/api/segment", { method: "POST", body: form });
}

/** Status of a job; `statusUrl` is the absolute URL the API returned. */
export async function getJob(statusUrl: string, signal?: AbortSignal): Promise<JobStatus> {
  const path = statusUrl.startsWith(API_URL) ? statusUrl.slice(API_URL.length) : new URL(statusUrl).pathname;
  return request(path, { signal, cache: "no-store" });
}

export function getSampleStudy(): Promise<SampleStudy> {
  return request("/api/sample-study");
}

/**
 * The sample study's segmentation, pre-computed on the server: no files are sent or downloaded.
 * Fails with code `sample_result_unavailable` while it is still being computed, or where there is none.
 */
export function segmentSampleStudy(): Promise<JobAccepted & { case: string }> {
  return request("/api/sample-study/segment", { method: "POST" });
}

/** The static demo's sample study result. Its `report_url` is a folder of `<plane>_<sequence>.png`. */
export function getDemoSampleStudy(): Promise<SegmentationResult> {
  return demoFile("/segment/result.json");
}

// ---- Model card ----

export type PairedComparison = {
  difference: number;
  ci95: [number, number];
  patients_better: number;
  patients_worse: number;
};

export type ModelCard = {
  headline: {
    served_accuracy_at_128px: number;
    original_accuracy_at_128px: number;
    served_accuracy: number;
    original_accuracy: number;
    mcnemar_p: number | null;
  };
  specification: {
    segmentation: {
      patients: number;
      test_patients: number;
      folds: number;
      parameters: number | null;
      /** Edge of the cubic sliding window, or one size per axis. */
      patch_size: number | number[] | null;
      et_min_voxels: number;
    };
    classification: {
      images: number;
      split: { train: number; val: number; test: number };
      parameters: number | null;
      image_size: number | null;
    };
  };
  segmentation: {
    test_patients: number;
    single_best_fold: string;
    rows: { model: string; post_processed: boolean; served: boolean; WT: number; TC: number; ET: number; mean: number }[];
    ensemble_vs_single_best: PairedComparison;
    et_cleanup_on_ensemble: PairedComparison;
    patients_without_et: number;
    patients_without_et_scored_1: { raw: number; post_processed: number };
    cross_validation: { mean: number; std: number; folds: number };
  };
  classification: {
    test_images: number;
    models: {
      key: string;
      name: string;
      change: string;
      served: boolean;
      checkpoint_epoch: number;
      accuracy: number;
      macro_f1: number;
      accuracy_at_128px: number;
    }[];
  };
  stress_test: {
    tumour_images: number;
    conditions: { key: string; label: string }[];
    models: { key: string; name: string; served: boolean; values: number[]; accuracy: number[] }[];
  };
  limitations: { title: string; body: string }[];
  sources: string[];
};

export async function getModelCard(): Promise<ModelCard> {
  if (STATIC_DEMO) return (await import("../../public/demo/model-card.json")).default as unknown as ModelCard;
  return request("/api/results/model-card");
}
