import type { Metadata } from "next";
import { Suspense } from "react";

import { SegmentScreen } from "@/components/segment/SegmentScreen";

export const metadata: Metadata = { title: "Segment" };

export default function SegmentPage() {
  // The screen reads the URL (?demo=3d), which is only known at request time.
  return (
    <Suspense>
      <SegmentScreen />
    </Suspense>
  );
}
