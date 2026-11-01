import type { Metadata } from "next";

import { SegmentScreen } from "@/components/segment/SegmentScreen";

export const metadata: Metadata = { title: "Segment" };

export default function SegmentPage() {
  return <SegmentScreen />;
}
