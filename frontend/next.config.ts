import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  distDir: process.env.YAMI_DIST || ".next",
};

export default nextConfig;
