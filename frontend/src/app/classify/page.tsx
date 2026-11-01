import type { Metadata } from "next";

import { ClassifyScreen } from "@/components/classify/ClassifyScreen";

export const metadata: Metadata = { title: "Classify" };

export default function ClassifyPage() {
  return <ClassifyScreen />;
}
