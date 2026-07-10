/**
 * App bootstrap: view flow (onboarding wall → dashboard) + the search
 * pipeline. First visit shows the ❤️/🥔/⏭️ wall; once ≥10 reactions are
 * in localStorage the dashboard (CF picks) and the search bar appear.
 */

import "./style.css";

import { applyBlend, cfNudges, DEFAULT_ALPHA, type Nudge } from "./blend";
import { loadCatalog } from "./data";
import { renderDashboard } from "./dashboard";
import { loadNeighbors } from "./neighbors";
import { getRatings, ratingCount } from "./ratings";
import { search } from "./search";
import type { SearchResult } from "./types";
import { renderGrid, searchStatus, setStatus } from "./ui";
import { renderWall } from "./wall";

const queryEl = document.getElementById("query") as HTMLInputElement;
const goEl = document.getElementById("go") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLElement;
const resultsEl = document.getElementById("results") as HTMLElement;
const emptyEl = document.getElementById("empty") as HTMLElement;
const onboardingEl = document.getElementById("onboarding") as HTMLElement;
const dashboardEl = document.getElementById("dashboard") as HTMLElement;
const searchboxEl = document.getElementById("searchbox") as HTMLElement;

const GATE = 10;
let busy = false;

const debugEl = document.getElementById("debug") as HTMLDetailsElement;
const alphaEl = document.getElementById("alpha") as HTMLInputElement;
const alphaOut = document.getElementById("alpha-out") as HTMLOutputElement;

/** Last search, kept so the α slider can re-blend without re-searching. */
let lastResults: SearchResult[] = [];
let lastNudges: Map<number, Nudge> = new Map();
let lastMode: Parameters<typeof searchStatus>[1] = "hybrid";

/**
 * Current blend weight. Pre-onboarding (no ratings) the blend is forced
 * OFF (α=0, pure search) regardless of the slider — there is no taste
 * signal to apply.
 */
function currentAlpha(): number {
  return ratingCount() === 0 ? 0 : Number(alphaEl.value);
}

/**
 * Blend (or re-blend) the cached results at the current α and render.
 * Called after every search and on every slider move — the live
 * reorder IS the blend demonstration (no offline ground truth exists;
 * see EVALUATION.md §3).
 */
function renderBlended(): void {
  const alpha = currentAlpha();
  const shown = applyBlend(lastResults, lastNudges, alpha);
  renderGrid(resultsEl, shown, alpha > 0 ? lastNudges : undefined);
  setStatus(statusEl, searchStatus(shown.length, lastMode, alpha));
  emptyEl.hidden = shown.length > 0;
}

/**
 * Show the onboarding wall view (dashboard + search hidden).
 * Used on first visit and via the dashboard's "rate more" button.
 */
async function showOnboarding(): Promise<void> {
  dashboardEl.hidden = true;
  searchboxEl.hidden = true;
  resultsEl.hidden = true;
  onboardingEl.hidden = false;
  renderWall(onboardingEl, await loadCatalog(), showDashboard);
}

/**
 * Show the dashboard view (wall hidden, search available below).
 * Re-renders picks from the current localStorage ratings.
 */
async function showDashboard(): Promise<void> {
  onboardingEl.hidden = true;
  dashboardEl.hidden = false;
  searchboxEl.hidden = false;
  resultsEl.hidden = false;
  await renderDashboard(dashboardEl, await loadCatalog(), () => void showOnboarding());
}

/**
 * Run one search interaction end-to-end: read the query, show progress,
 * render the grid (or the sad potato), surface degraded mode.
 *
 * Serialized via `busy` — a second submit while one is in flight is
 * ignored rather than queued (last-write-wins UIs confuse more than
 * they help at this scale).
 */
async function onSearch(): Promise<void> {
  const query = queryEl.value.trim();
  if (!query || busy) return;
  busy = true;
  goEl.disabled = true;
  emptyEl.hidden = true;
  setStatus(statusEl, "searching…");

  try {
    const { results, mode } = await search(query);
    lastResults = results;
    lastMode = mode;
    lastNudges = new Map();
    if (ratingCount() > 0 && results.length > 0) {
      try {
        const [neighbors, catalog] = await Promise.all([loadNeighbors(), loadCatalog()]);
        const titleOf = new Map(catalog.map((m) => [m.id, m.title]));
        lastNudges = cfNudges(results, neighbors, getRatings(), titleOf);
      } catch (e) {
        console.warn("blend unavailable (neighbors failed to load):", e);
      }
    }
    debugEl.hidden = ratingCount() === 0;
    renderBlended();
  } catch (e) {
    console.error(e);
    setStatus(statusEl, `<span class="warn">search failed — try again</span>`);
  } finally {
    busy = false;
    goEl.disabled = false;
  }
}

/**
 * Pre-warm the embedding model in the background after first paint so
 * the first search doesn't eat the full model download; progress shows
 * in the status line. Dynamically imported so transformers.js stays out
 * of the entry chunk. Failures are silent — search still works
 * (BM25-only) without the model.
 */
function prewarm(): void {
  const popcorn = `<img class="mini" src="${import.meta.env.BASE_URL}popcorn.png" alt="" />`;
  setStatus(statusEl, `${popcorn} warming up semantic search…`);
  import("./embed")
    .then(({ getExtractor }) =>
      getExtractor((frac) =>
        setStatus(statusEl, `${popcorn} loading embedding model… ${Math.round(frac * 100)}%`),
      ),
    )
    .then(() => setStatus(statusEl, ""))
    .catch(() => setStatus(statusEl, ""));
}

goEl.addEventListener("click", onSearch);
queryEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") void onSearch();
});
alphaEl.value = String(DEFAULT_ALPHA);
alphaEl.addEventListener("input", () => {
  alphaOut.textContent = Number(alphaEl.value).toFixed(2);
  if (lastResults.length > 0) renderBlended();
});
prewarm();
void (ratingCount() >= GATE ? showDashboard() : showOnboarding());
