/** Shared data shapes for the movieBro client. */

/** One catalog movie (a row of movies.json). */
export interface Movie {
  /** MovieLens movieId — the canonical id everywhere. */
  id: number;
  /** Display title in natural order ("The Matrix"). */
  title: string;
  /** Release year string; "" when unknown. */
  year: string;
  /** MovieLens genre labels. */
  genres: string[];
  /** TMDB poster path ("/xyz.jpg") or null for unenriched movies. */
  poster: string | null;
  /** Rating count in ml-latest-small — the popularity signal. */
  numRatings: number;
}

/** Child type emitted by build_index.py. */
export type ChildType =
  | "title"
  | "overview"
  | "tagline"
  | "keywords"
  | "tags"
  | "cast";

/** One search child: its array index in search-index.json is its id. */
export interface Child {
  /** Parent movie id. */
  movieId: number;
  /** What kind of text this is (drives provenance display). */
  type: ChildType;
  /** The searchable text itself. */
  text: string;
}

/** One retrieval leg's ranked result. */
export interface LegHit {
  /** Child id (index into the children array). */
  childId: number;
  /** Leg-native score (BM25 score or cosine similarity). */
  score: number;
}

/** A movie-level search result after fusion + grouping. */
export interface SearchResult {
  /** The matched movie. */
  movie: Movie;
  /** Fused (RRF) score of the movie's best child. */
  score: number;
  /** The single best-matching child (provenance snippet). */
  bestChild: Child;
  /** Which legs contributed this movie's best child. */
  legs: { dense: boolean; bm25: boolean };
}
