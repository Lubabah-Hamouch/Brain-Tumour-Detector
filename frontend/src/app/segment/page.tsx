import type { Metadata } from "next";

import { PlannedScreen } from "@/components/PlannedScreen";

export const metadata: Metadata = { title: "Segment" };

export default function SegmentPage() {
  return (
    <PlannedScreen
      label="Segmentation"
      title="Segmentation workspace"
      body="Upload, live job progress and the results viewer are built next, on top of POST /api/segment and GET /api/jobs/{id}."
    />
  );
}
