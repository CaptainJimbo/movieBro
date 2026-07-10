/**
 * Onboarding poster wall: a popular-but-diverse seed set the user
 * reacts to with ❤️ / 🥔 / ⏭️. Gate: ≥10 real reactions (skips don't
 * count), open ceiling — more is better. ⏭️ swaps in the next
 * recognizable title from the same genre queue so the wall stays
 * productive.
 */

import { POSTER_BASE } from "./config";
import { addSkip, getRatings, getSkips, ratingCount, setRating } from "./ratings";
import type { Movie } from "./types";

const WALL_SIZE = 18;
const GATE = 10;

/** Per-genre popularity-sorted queues feeding the wall. */
interface SeedQueues {
  genres: string[];
  queues: Map<string, Movie[]>;
  cursor: Map<string, number>;
}

/**
 * Build stratified seed queues: each movie files under its primary
 * (first-listed) genre; queues are popularity-sorted so swapped-in
 * posters stay recognizable. Already rated/skipped movies are excluded.
 *
 * @param movies - full catalog.
 * @returns Queues + round-robin genre order (most-populated first).
 */
function buildQueues(movies: Movie[]): SeedQueues {
  const rated = getRatings();
  const skipped = getSkips();
  const queues = new Map<string, Movie[]>();
  for (const m of movies) {
    if (!m.poster || rated[m.id] !== undefined || skipped.has(m.id)) continue;
    const g = m.genres[0];
    if (!g || g === "(no genres listed)") continue;
    if (!queues.has(g)) queues.set(g, []);
    queues.get(g)!.push(m);
  }
  for (const q of queues.values()) q.sort((a, b) => b.numRatings - a.numRatings);
  const genres = [...queues.keys()].sort(
    (a, b) => queues.get(b)!.length - queues.get(a)!.length,
  );
  return { genres, queues, cursor: new Map(genres.map((g) => [g, 0])) };
}

/**
 * Pull the next unseen movie from a genre queue.
 *
 * @param seeds - the queue state (cursor advances).
 * @param genre - which queue.
 * @returns The next movie, or null when the queue is exhausted.
 */
function nextFrom(seeds: SeedQueues, genre: string): Movie | null {
  const q = seeds.queues.get(genre) ?? [];
  const i = seeds.cursor.get(genre) ?? 0;
  seeds.cursor.set(genre, i + 1);
  return q[i] ?? null;
}

/**
 * Render the onboarding wall into a container and manage its lifecycle.
 *
 * @param container - the #onboarding element.
 * @param movies - full catalog.
 * @param onGateOpen - called when the user crosses GATE reactions AND
 *   clicks the unlocked "show my dashboard" button.
 */
export function renderWall(
  container: HTMLElement,
  movies: Movie[],
  onGateOpen: () => void,
): void {
  const seeds = buildQueues(movies);
  container.innerHTML = `
    <img class="banner" src="${import.meta.env.BASE_URL}banner.png"
         alt="Rate 10, I'll do the rest" />
    <p class="wall-hint">React to movies you've seen — ❤️ loved it, 🥔 dud,
      ⏭️ haven't seen it. <strong>10 unlocks your dashboard; more = sharper
      picks.</strong></p>
    <div class="wall-progress"><span id="wall-count"></span><button id="wall-done" hidden>
      Show my dashboard →</button></div>
    <div class="wall-grid" id="wall-grid"></div>`;

  const grid = container.querySelector<HTMLElement>("#wall-grid")!;
  const countEl = container.querySelector<HTMLElement>("#wall-count")!;
  const doneBtn = container.querySelector<HTMLButtonElement>("#wall-done")!;
  doneBtn.addEventListener("click", onGateOpen);

  /** Refresh the "n rated" progress line and gate button visibility. */
  function updateProgress(): void {
    const n = ratingCount();
    countEl.textContent =
      n < GATE ? `${n}/${GATE} rated — keep going`
               : `${n} rated — nice, more is better`;
    doneBtn.hidden = n < GATE;
  }

  /**
   * Render one wall card; its buttons replace the card in place (⏭️ and
   * after-rating both pull from the same genre queue so the wall stays
   * full).
   */
  function makeCard(movie: Movie, genre: string): HTMLElement {
    const card = document.createElement("article");
    card.className = "wallcard";
    card.innerHTML = `
      <img src="${POSTER_BASE}${movie.poster}" alt="" loading="lazy" />
      <div class="wall-title">${movie.title}${movie.year ? ` (${movie.year})` : ""}</div>
      <div class="wall-actions">
        <button data-act="love" title="loved it">❤️</button>
        <button data-act="nope" title="dud">🥔</button>
        <button data-act="skip" title="haven't seen — show another">⏭️</button>
      </div>`;
    card.querySelectorAll<HTMLButtonElement>("button").forEach((b) =>
      b.addEventListener("click", () => {
        const act = b.dataset.act!;
        if (act === "love") setRating(movie.id, 1);
        else if (act === "nope") setRating(movie.id, -1);
        else addSkip(movie.id);
        const next = nextFrom(seeds, genre);
        if (next) card.replaceWith(makeCard(next, genre));
        else card.remove();
        updateProgress();
      }),
    );
    return card;
  }

  for (let i = 0; i < WALL_SIZE; i++) {
    const genre = seeds.genres[i % seeds.genres.length];
    const m = nextFrom(seeds, genre);
    if (m) grid.appendChild(makeCard(m, genre));
  }
  updateProgress();
}
