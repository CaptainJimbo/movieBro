/**
 * "Your next watch" dashboard: one CF pick per top-3 genre with
 * "because you ❤️ X, Y" provenance, ❤️/🥔 on every card, and the
 * "more like this" item-item strip ("viewers also liked").
 *
 * Ratings made here re-run the fold-in ONLY behind the explicit
 * "refresh picks" button — no violent mid-session reshuffles (spec).
 */

import { POSTER_BASE } from "./config";
import { genrePicks, nextPickForGenre, type Pick } from "./foldin";
import { loadNeighbors, type NeighborMap } from "./neighbors";
import { getRatings, setRating } from "./ratings";
import type { Movie } from "./types";

const STRIP_SIZE = 8;

/**
 * Render (or re-render) the dashboard into its container.
 *
 * @param container - the #dashboard element.
 * @param movies - full catalog.
 * @param onRateMore - navigates back to the onboarding wall.
 * @param onMyRatings - navigates to the ratings manager view.
 */
export async function renderDashboard(
  container: HTMLElement,
  movies: Movie[],
  onRateMore: () => void,
  onMyRatings: () => void,
): Promise<void> {
  container.innerHTML = `<p class="dim">crunching your taste…</p>`;
  const neighbors = await loadNeighbors();
  const byId = new Map(movies.map((m) => [m.id, m]));
  const picks = genrePicks(movies, neighbors, getRatings());

  container.innerHTML = `
    <div class="dash-head">
      <h2>Your next watch</h2>
      <span class="dash-tools">
        <button id="refresh-picks" title="re-run recommendations with your latest ratings">↻ refresh picks</button>
        <button id="rate-more">＋ rate more movies</button>
        <button id="my-ratings">❤️ my ratings</button>
      </span>
    </div>
    <div class="picks" id="picks"></div>
    <div class="strip" id="strip" hidden></div>`;

  container.querySelector("#rate-more")!.addEventListener("click", onRateMore);
  container.querySelector("#my-ratings")!.addEventListener("click", onMyRatings);
  container.querySelector("#refresh-picks")!.addEventListener("click", () => {
    void renderDashboard(container, movies, onRateMore, onMyRatings);
  });

  const picksEl = container.querySelector<HTMLElement>("#picks")!;
  const stripEl = container.querySelector<HTMLElement>("#strip")!;

  if (picks.length === 0) {
    picksEl.innerHTML = `<p class="dim">Not enough signal yet — rate a few
      more movies (❤️ especially) and hit refresh.</p>`;
    return;
  }

  /** Movie ids currently displayed as picks (replacements must differ). */
  const onScreen = new Set(picks.map(([, p]) => p.movie.id));

  /**
   * Build one pick card. Rating it swaps ONLY this genre bin: the fold-in
   * reruns with the fresh rating and the next best candidate in the same
   * genre slides in (other bins untouched — no global reshuffle).
   */
  function makePickCard(genre: string, pick: Pick): HTMLElement {
    const m = pick.movie;
    const because = pick.because.map((b) => b.title).join(", ");
    const card = document.createElement("article");
    card.className = "pickcard";
    card.innerHTML = `
      <div class="pick-label">Top pick in <strong>${genre}</strong></div>
      ${m.poster ? `<img src="${POSTER_BASE}${m.poster}" alt="" loading="lazy" />`
                 : `<div class="noposter">${m.title}</div>`}
      <div class="pick-body">
        <div class="pick-title">${m.title}${m.year ? ` (${m.year})` : ""}</div>
        ${because ? `<div class="pick-why">because you ❤️ ${because}</div>` : ""}
        <div class="pick-actions">
          <button data-act="love" title="loved it" aria-label="Love ${m.title}">❤️</button>
          <button data-act="nope" title="dud" aria-label="Nope ${m.title}">🥔</button>
          <button data-act="more" title="viewers also liked" aria-label="More movies like ${m.title}">more like this ▸</button>
        </div>
      </div>`;

    card.querySelectorAll<HTMLButtonElement>("button").forEach((b) =>
      b.addEventListener("click", () => {
        const act = b.dataset.act!;
        if (act === "more") {
          renderStrip(stripEl, m, neighbors, byId);
          return;
        }
        setRating(m.id, act === "love" ? 1 : -1);
        onScreen.delete(m.id); // rated movies are excluded by fold-in anyway
        const next = nextPickForGenre(genre, movies, neighbors, getRatings(), onScreen);
        if (next) {
          onScreen.add(next.movie.id);
          card.replaceWith(makePickCard(genre, next));
        } else {
          card.innerHTML = `<div class="pick-body"><p class="dim">No more
            strong picks in ${genre} yet — rate more movies.</p></div>`;
        }
      }),
    );
    return card;
  }

  for (const [genre, pick] of picks) picksEl.appendChild(makePickCard(genre, pick));
}

/**
 * Render the "viewers also liked" strip for one movie — pure item-item
 * CF from neighbors.json, NOT personalized, and deliberately outside
 * the search grid (it answers "similar to this movie", not the query).
 *
 * @param stripEl - the #strip element.
 * @param movie - the anchor movie.
 * @param neighbors - the CF model.
 * @param byId - catalog lookup.
 */
function renderStrip(
  stripEl: HTMLElement,
  movie: Movie,
  neighbors: NeighborMap,
  byId: Map<number, Movie>,
): void {
  const list = (neighbors.get(movie.id) ?? [])
    .map(([nid]) => byId.get(nid))
    .filter((m): m is Movie => !!m && !!m.poster)
    .slice(0, STRIP_SIZE);

  stripEl.hidden = false;
  stripEl.innerHTML = `
    <h3>Viewers who liked <em>${movie.title}</em> also liked</h3>
    <div class="strip-row">
      ${list.map((m) => `
        <figure class="stripcard" title="${m.title}${m.year ? ` (${m.year})` : ""}">
          <img src="${POSTER_BASE}${m.poster}" alt="" loading="lazy" />
          <figcaption>${m.title}</figcaption>
        </figure>`).join("")}
    </div>`;
  stripEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
