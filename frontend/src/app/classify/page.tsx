import type { Metadata } from "next";

import { PlannedScreen } from "@/components/PlannedScreen";

export const metadata: Metadata = { title: "Classify" };

export default function ClassifyPage() {
  return (
    <PlannedScreen
      label="Classification"
      title="Tumour-type classification"
      body="Upload and the prediction view with Grad-CAM are built next, on top of POST /api/classify."
    />
  );
}
