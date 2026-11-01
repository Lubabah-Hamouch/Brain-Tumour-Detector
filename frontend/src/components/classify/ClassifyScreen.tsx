"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  ApiUnavailable,
  classify,
  getResultsSummary,
  getSamples,
  type ClassificationResult,
  type ResultsSummary,
  type Sample,
} from "@/lib/api";
import { checkImage } from "@/lib/imageCheck";

import { ResultView } from "./ResultView";
import type { Problem, Selected } from "./types";
import { UploadView } from "./UploadView";

/**
 * Classification screen: upload (or pick a test-set sample) -> checked preview -> result.
 * One component owns the state so the result view can start a new run without a page change.
 */
export function ClassifyScreen() {
  const [selected, setSelected] = useState<Selected | null>(null);
  const [problem, setProblem] = useState<Problem | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  // A result is kept together with the file it belongs to.
  const [done, setDone] = useState<{ result: ClassificationResult; target: Selected } | null>(null);
  const [samples, setSamples] = useState<Sample[] | null>(null);
  const [samplesError, setSamplesError] = useState<string | null>(null);
  const [summary, setSummary] = useState<ResultsSummary | null>(null);
  const previewUrl = useRef<string | null>(null);

  useEffect(() => {
    getSamples().then(setSamples, (e: Error) => setSamplesError(e.message));
    // Only used for context (model line, runner-up note); the screen works without it.
    getResultsSummary().then(setSummary, () => undefined);
  }, []);

  useEffect(() => () => {
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
  }, []);

  const run = useCallback(async (target: Selected) => {
    setRunning(true);
    setRunError(null);
    try {
      const response = await classify(target.file);
      setDone({ result: response, target });
      window.scrollTo({ top: 0 });
    } catch (error) {
      setDone(null);
      setRunError(error instanceof ApiError || error instanceof ApiUnavailable ? error.message : String(error));
    } finally {
      setRunning(false);
    }
  }, []);

  /** Check a file; on success it becomes the selection. Returns the selection, or null. */
  const accept = useCallback(async (file: File, sample?: Sample) => {
    setRunError(null);
    const check = await checkImage(file);
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    previewUrl.current = check.ok ? check.previewUrl : null;
    if (!check.ok) {
      setSelected(null);
      setProblem({ name: file.name, bytes: file.size, message: check.message });
      return null;
    }
    const next: Selected = { file, info: check.info, previewUrl: check.previewUrl, sample };
    setSelected(next);
    setProblem(null);
    return next;
  }, []);

  const loadSample = useCallback(
    async (sample: Sample) => {
      try {
        const response = await fetch(sample.url);
        if (!response.ok) throw new Error(`The sample could not be loaded (${response.status}).`);
        const blob = await response.blob();
        return accept(new File([blob], sample.name, { type: blob.type || "image/jpeg" }), sample);
      } catch (error) {
        setSelected(null);
        setProblem({ name: sample.name, bytes: sample.size_bytes, message: (error as Error).message });
        return null;
      }
    },
    [accept],
  );

  /** From the result view: a new file or sample goes straight to classification when it passes the checks. */
  const rerun = useCallback(
    async (next: Promise<Selected | null>) => {
      const target = await next;
      if (target) await run(target);
      else setDone(null);
    },
    [run],
  );

  const reset = useCallback(() => {
    setDone(null);
    setSelected(null);
    setProblem(null);
    setRunError(null);
  }, []);

  if (done) {
    return (
      <ResultView
        key={done.result.result_id}
        result={done.result}
        selected={done.target}
        summary={summary}
        samples={samples}
        running={running}
        onFile={(file) => rerun(accept(file))}
        onSample={(sample) => rerun(loadSample(sample))}
        onReset={reset}
      />
    );
  }

  return (
    <UploadView
      selected={selected}
      problem={problem}
      running={running}
      runError={runError}
      samples={samples}
      samplesError={samplesError}
      onFile={accept}
      onSample={(sample) => rerun(loadSample(sample))}
      onClear={reset}
      onClassify={() => selected && run(selected)}
    />
  );
}
