import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `base: "/metrics-app/"` so that the production build's asset URLs match the
// Python server's static mount at /metrics-app/*. Dev server still runs at /.
export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === "build" ? "/metrics-app/" : "/",
  server: {
    port: 5174,
    // Proxy API calls to the Python server during dev so the SPA can fetch
    // /api/projects/:slug/runs/:run_id and /artifact/:key without CORS.
    proxy: {
      "/api": "http://127.0.0.1:8765",
    },
  },
}));
