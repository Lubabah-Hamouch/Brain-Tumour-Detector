/**
 * Typed access to the FastAPI backend (backend/). Server components call it with
 * API_URL; the browser uses NEXT_PUBLIC_API_URL. Both default to the local server.
 */

export const API_URL = process.env.API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8010";

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
    resolution_note: string;
  };
  model: { checkpoint: string; checkpoint_epoch: number };
  warnings: string[];
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

export function getResultsSummary(): Promise<ResultsSummary> {
  return request("/api/results/summary");
}

export function getSamples(): Promise<Sample[]> {
  return request("/api/samples");
}

export function classify(file: File): Promise<ClassificationResult> {
  const form = new FormData();
  form.append("file", file);
  return request("/api/classify", { method: "POST", body: form });
}
