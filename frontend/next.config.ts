import path from "node:path";

import type { NextConfig } from "next";

// This folder is the app's root. Pinning it stops Next.js from guessing a workspace root from
// lockfiles in parent directories.
const appRoot = path.resolve(__dirname);

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  outputFileTracingRoot: appRoot,
  turbopack: { root: appRoot },
  // Keep the dev server from writing extra files into the project.
  agentRules: false,
};

export default nextConfig;
