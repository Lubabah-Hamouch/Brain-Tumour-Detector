import path from "node:path";

import type { NextConfig } from "next";

// This folder is the app's root. Pinning it stops Next.js from guessing a workspace root from
// lockfiles in parent directories.
const appRoot = path.resolve(__dirname);

// The browser's API address is fixed into the client bundle at build time. A production build
// without it is the static demo (src/lib/api.ts): say so in the build log, so a forgotten variable
// is not mistaken for a choice.
if (process.env.NODE_ENV === "production" && !process.env.NEXT_PUBLIC_API_URL) {
  console.log(
    "NEXT_PUBLIC_API_URL is not set: building the static demo (samples only, from public/demo/). " +
      "Set it to the API's URL for uploads; see DEPLOYMENT.md.",
  );
}

const nextConfig: NextConfig = {
  // A self-contained server in .next/standalone, used by the Docker image. Vercel packages the app
  // its own way.
  output: process.env.VERCEL ? undefined : "standalone",
  cacheComponents: true,
  partialPrefetching: true,
  outputFileTracingRoot: appRoot,
  turbopack: { root: appRoot },
  // Keep the dev server from writing extra files into the project.
  agentRules: false,
};

export default nextConfig;
