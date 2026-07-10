import { defineConfig } from "vite";

/**
 * Vite config. `base` matches the GitHub Pages project path
 * (captainjimbo.github.io/movieBro/) so built asset URLs resolve; dev
 * server ignores it for local work.
 */
export default defineConfig({
  base: "/movieBro/",
});
