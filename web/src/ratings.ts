/**
 * The user's taste store — localStorage only. NOTHING leaves the device:
 * no server sees a rating, a skip, or a derived score (the Worker only
 * ever receives anonymous query vectors).
 *
 * Scale: ❤️ = +1, 🥔 = −1. Skips (⏭️ "haven't seen") are stored
 * separately and carry NO rating signal — they only stop the onboarding
 * wall from re-showing a poster.
 */

const RATINGS_KEY = "moviebro-ratings-v1";
const SKIPS_KEY = "moviebro-skips-v1";

/** movieId -> +1 (❤️) or −1 (🥔). */
export type Ratings = Record<number, 1 | -1>;

/**
 * Read all ratings from localStorage.
 *
 * @returns The ratings map ({} on first visit or parse failure).
 */
export function getRatings(): Ratings {
  try {
    return JSON.parse(localStorage.getItem(RATINGS_KEY) ?? "{}") as Ratings;
  } catch {
    return {};
  }
}

/**
 * Record (or overwrite) one rating and persist.
 *
 * @param movieId - the rated movie.
 * @param value - +1 for ❤️, −1 for 🥔.
 */
export function setRating(movieId: number, value: 1 | -1): void {
  const r = getRatings();
  r[movieId] = value;
  localStorage.setItem(RATINGS_KEY, JSON.stringify(r));
}

/**
 * Count real reactions (❤️/🥔 — the onboarding gate currency; skips
 * never count).
 *
 * @returns Number of rated movies.
 */
export function ratingCount(): number {
  return Object.keys(getRatings()).length;
}

/**
 * Read the skipped-movie set (onboarding ⏭️ history).
 *
 * @returns Set of movieIds the user marked "haven't seen".
 */
export function getSkips(): Set<number> {
  try {
    return new Set(JSON.parse(localStorage.getItem(SKIPS_KEY) ?? "[]") as number[]);
  } catch {
    return new Set();
  }
}

/**
 * Add one movie to the skip set and persist.
 *
 * @param movieId - the skipped movie.
 */
export function addSkip(movieId: number): void {
  const s = getSkips();
  s.add(movieId);
  localStorage.setItem(SKIPS_KEY, JSON.stringify([...s]));
}
