import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone output: bundles a minimal server + only the deps actually
  // used, so the Docker image doesn't need the full node_modules tree.
  output: "standalone",
};

export default nextConfig;
