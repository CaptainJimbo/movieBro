/**
 * "My ratings" view: everything the user has ❤️'d or 🥔'd, amendable in
 * place — flip a rating, or remove it entirely (✕ makes the movie
 * unrated again: recommendable, zero fold-in signal). Two labeled
 * sections keep loved/noped scannable. All edits hit localStorage
 * immediately; the dashboard re-derives picks next time it renders.
 */

import { POSTER_BASE } from "./config";
import { getRatings, removeRating, setRating } from "./ratings";
import type { Movie } from "./types";

/**
 * Render the ratings manager into its container.
 *
 * @param container - the #myratings element.
 * @param movies - full catalog (id -> poster/title resolution).
 * @param onBack - navigates back to the dashboard.
 */
export function renderMyRatings(
  container: HTMLElement,
  movies: Movie[],
  onBack: () => void,
): void {
  const byId = new Map(movies.map((m) => [m.id, m]));
  const ratings = getRatings();
  const loved: Movie[] = [];
  const noped: Movie[] = [];
  for (const [mid, r] of Object.entries(ratings)) {
    const m = byId.get(Number(mid));
    if (m) (r > 0 ? loved : noped).push(m);
  }
  const byTitle = (a: Movie, b: Movie) => a.title.localeCompare(b.title);
  loved.sort(byTitle);
  noped.sort(byTitle);

  container.innerHTML = `
    <div class="dash-head">
      <h2>My ratings</h2>
      <span class="dash-tools"><button id="ratings-back">← dashboard</button></span>
    </div>
    <p class="dim">${loved.length} ❤️ · ${noped.length} 🥔 — flip a rating or
      ✕ to remove it (removed movies become recommendable again).</p>
    <div id="ratings-sections"></div>`;
  container.querySelector("#ratings-back")!.addEventListener("click", onBack);
  const sections = container.querySelector<HTMLElement>("#ratings-sections")!;

  /**
   * Build one rated-movie row: poster thumb, title, flip buttons with
   * the current rating highlighted, and remove. Edits update the store
   * and this row only (no full re-render).
   */
  function makeRow(movie: Movie): HTMLElement {
    const row = document.createElement("div");
    row.className = "raterow";

    /** Repaint the flip buttons to match the store's current value. */
    function paint(): void {
      const r = getRatings()[movie.id];
      row.querySelector('[data-act="love"]')!.classList.toggle("chosen", r === 1);
      row.querySelector('[data-act="nope"]')!.classList.toggle("chosen", r === -1);
    }

    row.innerHTML = `
      ${movie.poster ? `<img src="${POSTER_BASE}${movie.poster}" alt="" loading="lazy" />`
                     : `<div class="thumb-blank"></div>`}
      <span class="raterow-title">${movie.title}${movie.year ? ` (${movie.year})` : ""}</span>
      <span class="raterow-actions">
        <button data-act="love" title="loved it">❤️</button>
        <button data-act="nope" title="dud">🥔</button>
        <button data-act="remove" title="remove rating">✕</button>
      </span>`;
    row.querySelectorAll<HTMLButtonElement>("button").forEach((b) =>
      b.addEventListener("click", () => {
        const act = b.dataset.act!;
        if (act === "remove") {
          removeRating(movie.id);
          row.remove();
          return;
        }
        setRating(movie.id, act === "love" ? 1 : -1);
        paint();
      }),
    );
    paint();
    return row;
  }

  for (const [label, list] of [["❤️ Loved", loved], ["🥔 Noped", noped]] as const) {
    if (list.length === 0) continue;
    const h = document.createElement("h3");
    h.className = "raterow-heading";
    h.textContent = `${label} (${list.length})`;
    sections.appendChild(h);
    for (const m of list) sections.appendChild(makeRow(m));
  }
  if (loved.length + noped.length === 0) {
    sections.innerHTML = `<p class="dim">Nothing rated yet.</p>`;
  }
}
