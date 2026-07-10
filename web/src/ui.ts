/** DOM rendering: the 3x3 grid, provenance hovers, and status line. */

import { GRID_SIZE, POSTER_BASE } from "./config";
import type { SearchMode } from "./search";
import type { SearchResult } from "./types";

/** Human labels for child types shown in the provenance hover. */
const TYPE_LABEL: Record<string, string> = {
  title: "title",
  overview: "plot",
  tagline: "tagline",
  keywords: "keywords",
  tags: "community tags",
  cast: "cast",
};

/**
 * Render the top results into the 3x3 grid.
 *
 * Each card: poster (or a titled placeholder for the ~1% unenriched),
 * and a hover overlay showing WHY it matched — the best child snippet,
 * its type, and D/B leg badges. Provenance is the product's money-shot;
 * never render a bare poster without its "why".
 *
 * @param container - the #results grid element.
 * @param results - grouped results (only the first GRID_SIZE render).
 */
export function renderGrid(container: HTMLElement, results: SearchResult[]): void {
  container.innerHTML = "";
  for (const r of results.slice(0, GRID_SIZE)) {
    const card = document.createElement("article");
    card.className = "card";
    card.tabIndex = 0;

    const poster = r.movie.poster
      ? `<img src="${POSTER_BASE}${r.movie.poster}" alt="" loading="lazy" />`
      : `<div class="noposter">${escapeHtml(r.movie.title)}</div>`;

    const badges = [
      r.legs.dense ? `<span class="badge dense" title="matched by semantic search">D</span>` : "",
      r.legs.bm25 ? `<span class="badge bm25" title="matched by keyword search">B</span>` : "",
      `<span class="badge type">${TYPE_LABEL[r.bestChild.type] ?? r.bestChild.type}</span>`,
    ].join("");

    card.innerHTML = `
      ${poster}
      <div class="why">
        <div class="title">${escapeHtml(r.movie.title)}${r.movie.year ? ` (${r.movie.year})` : ""}</div>
        <div class="snippet">“${escapeHtml(truncate(r.bestChild.text, 140))}”</div>
        <div class="badges">${badges}</div>
      </div>`;
    container.appendChild(card);
  }
}

/**
 * Update the status line (loading progress, degraded-mode warnings).
 *
 * @param el - the #status element.
 * @param html - message (may contain markup); "" hides the line.
 */
export function setStatus(el: HTMLElement, html: string): void {
  el.hidden = html === "";
  el.innerHTML = html;
}

/**
 * Compose the post-search status: result count + retrieval-mode badge.
 *
 * @param n - number of results shown.
 * @param mode - which legs actually ran.
 * @returns Status HTML ("" when nothing special to say).
 */
export function searchStatus(n: number, mode: SearchMode): string {
  const degraded =
    mode === "bm25-only"
      ? ` <span class="warn">⚠ lexical-only mode (semantic search unreachable)</span>`
      : "";
  return n === 0 ? "" : `${n} matches${degraded}`;
}

/**
 * HTML-escape untrusted text (movie titles/snippets go through TMDB and
 * community tags — treat all of it as untrusted).
 *
 * @param s - raw text.
 * @returns Escaped text safe for innerHTML interpolation.
 */
function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

/**
 * Truncate text at a word boundary with an ellipsis.
 *
 * @param s - input text.
 * @param max - max characters before cutting.
 * @returns Original or truncated string ending in "…".
 */
function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, s.lastIndexOf(" ", max)) + "…";
}
