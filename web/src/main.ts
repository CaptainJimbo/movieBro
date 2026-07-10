/**
 * App bootstrap: view flow (onboarding wall → dashboard) + the search
 * pipeline. First visit shows the ❤️/🥔/⏭️ wall; once ≥10 reactions are
 * in localStorage the dashboard (CF picks) and the search bar appear.
 */

import "./style.css";

import { loadCatalog } from "./data";
import { renderDashboard } from "./dashboard";
import { ratingCount } from "./ratings";
import { search } from "./search";
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
    renderGrid(resultsEl, results);
    setStatus(statusEl, searchStatus(results.length, mode));
    emptyEl.hidden = results.length > 0;
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
  setStatus(statusEl, "warming up semantic search…");
  import("./embed")
    .then(({ getExtractor }) =>
      getExtractor((frac) =>
        setStatus(statusEl, `loading embedding model… ${Math.round(frac * 100)}%`),
      ),
    )
    .then(() => setStatus(statusEl, ""))
    .catch(() => setStatus(statusEl, ""));
}

goEl.addEventListener("click", onSearch);
queryEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") void onSearch();
});
prewarm();
void (ratingCount() >= GATE ? showDashboard() : showOnboarding());
