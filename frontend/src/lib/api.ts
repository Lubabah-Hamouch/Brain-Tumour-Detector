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
  };
  sources: string[];
};

export class ApiUnavailable extends Error {}

export async function getResultsSummary(): Promise<ResultsSummary> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/results/summary`);
  } catch (cause) {
    throw new ApiUnavailable(`The API at ${API_URL} is not reachable.`, { cause });
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiUnavailable(body?.error?.message ?? `The API returned ${response.status}.`);
  }
  return response.json();
}
