import path from "node:path";

import type { NextConfig } from "next";

// This folder is the app's root. Pinning it stops Next.js from guessing a workspace root from
// lockfiles in parent directories.
const appRoot = path.resolve(__dirname);

// Vercel sets VERCEL=1 during its builds. The browser's API address is fixed into the client bundle
// at build time, so a build without it would produce a site that cannot reach the API: stop it here.
if (process.env.VERCEL && !process.env.NEXT_PUBLIC_API_URL) {
  throw new Error(
    "NEXT_PUBLIC_API_URL is not set. Add it in the Vercel project under Settings > Environment Variables " +
      "(the API's URL, e.g. https://<user>-<space>.hf.space), then redeploy. See DEPLOYMENT.md.",
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
