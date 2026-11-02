"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiError,
  ApiUnavailable,
  getSampleStudy,
  segmentSampleStudy,
  startSegmentation,
  type SegmentationResult,
  type SequenceField,
} from "@/lib/api";
import { axialPreview } from "@/lib/nifti";
import { checkFile, detectSequence, evaluate, FIELD, type StudyFile } from "@/lib/studyFiles";

import { ProcessingView } from "./ProcessingView";
import { ResultsView } from "./ResultsView";
import type { Preview, Run } from "./types";
import { UploadView } from "./UploadView";

let nextId = 0;

/**
 * Segmentation workspace: four NIfTI files -> background job with live progress -> results.
 * The run's timing starts when the visitor presses Run: the upload and the server's checks are
 * measured here, the job's stages come from the job API.
 *
 * "Run the sample patient" asks the server for the sample study's pre-computed result: no files
 * are downloaded or uploaded (the public deployment does not serve the raw BraTS volumes), and the
 * result opens directly. Only where the server has no result yet (a local API still computing it)
 * does it fall back to loading the sample's files and running them like an upload.
 *
 * With ?demo=3d (the Overview's "View in 3D" link) the sample runs straight away and the result
 * opens in the 3D view.
 */
export function SegmentScreen() {
  const demo = useSearchParams().get("demo") === "3d";
  const router = useRouter();
  const demoStep = useRef<"idle" | "loading" | "started">("idle");
  // Set by "Run the sample patient": start the job as soon as the sample's files pass their checks.
  const runWhenReady = useRef(false);
  const [files, setFiles] = useState<StudyFile[]>([]);
  const [sampleCase, setSampleCase] = useState<string | null>(null);
  const [loadingSample, setLoadingSample] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [showResults, setShowResults] = useState(false);
  // Results open at the top, on the viewer, wherever the processing screen was scrolled to.
  useEffect(() => {
    if (showResults) window.scrollTo({ top: 0 });
  }, [showResults]);
  const runCounter = useRef(0);
  // Set while the current run is the sample's pre-computed result, which opens without a stop on
  // the processing screen.
  const openDirectly = useRef(false);

  const evaluation = useMemo(() => evaluate(files), [files]);

  const addFiles = useCallback((incoming: File[], replace = false) => {
    const entries: StudyFile[] = incoming.map((file) => ({
      id: `f${nextId++}`,
      file,
      detected: detectSequence(file.name),
      assigned: null,
      check: { kind: "pending" },
    }));
    setFiles((current) => (replace ? entries : [...current, ...entries]));
    if (!replace) setSampleCase(null);
    for (const entry of entries) {
      checkFile(entry.file).then((check) =>
        setFiles((current) => current.map((f) => (f.id === entry.id ? { ...f, check } : f))),
      );
    }
  }, []);

  const loadSample = useCallback(async () => {
    setLoadingSample(true);
    setRunError(null);
    try {
      const sample = await getSampleStudy();
      const loaded = await Promise.all(
        sample.files.map(async (f) => {
          const response = await fetch(f.url);
          if (!response.ok) throw new Error(`${f.name} could not be loaded (${response.status}).`);
          return new File([await response.blob()], f.name, { type: "application/gzip" });
        }),
      );
      addFiles(loaded, true);
      setSampleCase(sample.case);
    } catch (error) {
      setRunError(`The sample study could not be loaded: ${(error as Error).message}`);
    } finally {
      setLoadingSample(false);
    }
  }, [addFiles]);

  const start = useCallback(async () => {
    if (!evaluation.ready) return;
    const byField = Object.fromEntries(
      evaluation.slots.map((slot) => [FIELD[slot.name], slot.file!.file]),
    ) as Record<SequenceField, File>;
    const flair = evaluation.slots[0].file!;
    const header = flair.check.kind === "nifti" ? flair.check.header : null;
    const id = ++runCounter.current;
    const startedAt = performance.now();
    setRunError(null);
    setResult(null);
    setShowResults(false);
    openDirectly.current = false;
    setRun({ id, startedAt, header, files: byField, sample: sampleCase, statusUrl: null, uploadSeconds: null, preview: null });

    // The dimmed preview is read from the local FLAIR file while the upload runs.
    if (header) {
      axialPreview(flair.file, header)
        .then((preview: Preview | null) => setRun((r) => (r?.id === id ? { ...r, preview } : r)))
        .catch(() => undefined);
    }
    try {
      const accepted = await startSegmentation(byField);
      const uploadSeconds = (performance.now() - startedAt) / 1000;
      setRun((r) => (r?.id === id ? { ...r, statusUrl: accepted.status_url, uploadSeconds } : r));
    } catch (error) {
      setRun(null);
      setRunError(
        error instanceof ApiError || error instanceof ApiUnavailable
          ? error.message
          : `The upload failed: ${(error as Error).message}`,
      );
    }
  }, [evaluation, sampleCase]);

  // Fallback for a sample without a pre-computed result: run it once its files pass their checks.
  useEffect(() => {
    if (runWhenReady.current && sampleCase && evaluation.ready && !run) {
      runWhenReady.current = false;
      void start();
    }
  }, [evaluation.ready, run, start, sampleCase]);

  const runSample = useCallback(async () => {
    setLoadingSample(true);
    setRunError(null);
    const startedAt = performance.now();
    try {
      const accepted = await segmentSampleStudy();
      const id = ++runCounter.current;
      openDirectly.current = true;
      setResult(null);
      setShowResults(false);
      setRun({
        id,
        startedAt,
        header: null,
        files: null,
        sample: accepted.case,
        statusUrl: accepted.status_url,
        uploadSeconds: (performance.now() - startedAt) / 1000,
        preview: null,
      });
    } catch (error) {
      if (error instanceof ApiError && error.code === "sample_result_unavailable") {
        // No pre-computed result on this server yet: load the files and run them as an upload.
        runWhenReady.current = true;
        await loadSample();
      } else {
        setRunError(
          error instanceof ApiError || error instanceof ApiUnavailable
            ? error.message
            : `The sample could not be run: ${(error as Error).message}`,
        );
      }
    } finally {
      setLoadingSample(false);
    }
  }, [loadSample]);

  // Demo: run the sample once.
  useEffect(() => {
    if (demo && demoStep.current === "idle") {
      demoStep.current = "started";
      void runSample();
    }
  }, [demo, runSample]);

  const complete = useCallback(
    (r: SegmentationResult) => {
      setResult(r);
      if (demo || openDirectly.current) setShowResults(true);
    },
    [demo],
  );

  const backToFiles = useCallback(() => {
    openDirectly.current = false;
    setRun(null);
    setResult(null);
    setShowResults(false);
  }, []);

  const newStudy = useCallback(() => {
    backToFiles();
    setFiles([]);
    setSampleCase(null);
    setRunError(null);
    if (demo) router.replace("/segment");
    window.scrollTo({ top: 0 });
  }, [backToFiles, demo, router]);

  if (run && result && showResults) {
    return (
      <ResultsView
        key={result.result_id}
        result={result}
        run={run}
        onNewStudy={newStudy}
        initialView={demo ? "3d" : "axial"}
      />
    );
  }
  if (run) {
    return (
      <ProcessingView
        key={run.id}
        run={run}
        onComplete={complete}
        onShowResults={() => setShowResults(true)}
        onBack={backToFiles}
      />
    );
  }
  return (
    <UploadView
      files={files}
      evaluation={evaluation}
      sampleCase={sampleCase}
      loadingSample={loadingSample}
      runError={runError}
      onAdd={(list) => addFiles(list)}
      onAssign={(id, seq) => setFiles((fs) => fs.map((f) => (f.id === id ? { ...f, assigned: seq } : f)))}
      onRemove={(id) => setFiles((fs) => fs.filter((f) => f.id !== id))}
      onSample={runSample}
      onRun={start}
    />
  );
}
