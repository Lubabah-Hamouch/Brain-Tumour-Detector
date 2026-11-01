import type { Metadata } from "next";

import { PlannedScreen } from "@/components/PlannedScreen";

export const metadata: Metadata = { title: "Model card" };

export default function ModelCardPage() {
  return (
    <PlannedScreen
      label="Model card"
      title="Model card"
      body="Test results, the format stress test and the limitations are built next. Until then, see reports/final_results.md."
    />
  );
}
