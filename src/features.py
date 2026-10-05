"""Vectorised candidate generation (blocking) and pair features.

Every Source 1 / Source 2+3 text field is turned into sparse matrices once.
Candidates are selected with chunked sparse matrix products, and pair features
are computed for whole arrays of pairs at once (no per-pair Python loops except
the small prefix/abbreviation feature).
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import config

FEATURE_COLUMNS = [
    # business name
    "name_word_jaccard", "name_core_jaccard", "name_overlap_ratio", "name_shared_tokens",
    "name_char_cosine", "name_squash_cosine", "name_word_cosine", "name_exact",
    "name_core_exact", "name_length_ratio", "name_first_token_equal", "name_prefix_ratio",
    # address
    "addr_word_jaccard", "addr_overlap_ratio", "addr_shared_tokens", "addr_char_cosine",
    "addr_word_cosine", "addr_numeric_shared", "addr_numeric_jaccard", "addr_length_ratio",
    "addr_missing", "addr_exact",
    # country / source
    "same_country", "country_mismatch", "is_source3",
    # combined
    "combined_score", "name_x_addr", "shared_numeric_all",
    # position of the pair inside its Source 1 candidate list
    "cand_rank", "cand_gap_to_best", "cand_best_score", "cand_count",
]


def _vectorise(vectorizer, texts, n_rows):
    """Fit on all texts; an all-empty corpus yields an empty matrix instead of an error."""
    try:
        return vectorizer.fit_transform(texts).tocsr()
    except ValueError:
        from scipy import sparse
        return sparse.csr_matrix((n_rows, 1), dtype=np.float32)


class Representation:
    """Sparse text representations of Source 1 (left) and Source 2+3 (right)."""

    def __init__(self, left, right):
        self.n_left, self.n_right = len(left), len(right)
        n_all = self.n_left + self.n_right
        self.is_source3 = (right["source"].to_numpy() == "S3").astype(np.float32)

        def texts(column):
            return pd.concat([left[column], right[column]]).tolist()

        def split(matrix):
            return matrix[: self.n_left], matrix[self.n_left:]

        def tfidf(column, **kwargs):
            vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **kwargs)
            return split(_vectorise(vec, texts(column), n_all))

        def binary(column, **kwargs):
            vec = CountVectorizer(dtype=np.float32, binary=True, lowercase=False, **kwargs)
            return split(_vectorise(vec, texts(column), n_all))

        # cosine representations
        self.name_char = tfidf("name_core_sorted", analyzer="char_wb", ngram_range=(2, 4))
        self.name_squash = tfidf("name_squashed", analyzer="char", ngram_range=(3, 4))
        self.name_word = tfidf("name_core", token_pattern=r"\S+")
        self.addr_char = tfidf("address_sorted", analyzer="char_wb", ngram_range=(2, 4))
        self.addr_word = tfidf("business_address_clean", token_pattern=r"\S+")
        # binary representations (Jaccard / overlap / shared counts)
        self.name_bin = binary("business_name_clean", token_pattern=r"\S+")
        self.core_bin = binary("name_core", token_pattern=r"\S+")
        self.addr_bin = binary("business_address_clean", token_pattern=r"\S+")
        self.num_bin = binary("business_address_clean", token_pattern=r"\d+")

        # scalar helpers
        self.name_len = (left["business_name_clean"].str.len().to_numpy(np.float32),
                         right["business_name_clean"].str.len().to_numpy(np.float32))
        self.addr_len = (left["business_address_clean"].str.len().to_numpy(np.float32),
                         right["business_address_clean"].str.len().to_numpy(np.float32))
        self.name_full = (left["business_name_clean"].to_numpy(), right["business_name_clean"].to_numpy())
        self.core_tokens = ([s.split() for s in left["name_core"]],
                            [s.split() for s in right["name_core"]])
        self.country_codes = country_codes(left, right)


def country_codes(left, right):
    """Integer codes for country strings; unseen countries simply get their own code."""
    codes, _ = pd.factorize(pd.concat([left["country_clean"], right["country_clean"]]))
    return codes[: len(left)], codes[len(left):]


# ---------------------------------------------------------------------------
# BLOCKING
# ---------------------------------------------------------------------------
def _top_k_pairs(scores, k, min_score, row_offset):
    """(row, col) pairs of the k best columns per row whose score >= min_score."""
    k = min(k, scores.shape[1])
    if k <= 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    values = np.take_along_axis(scores, idx, axis=1)
    keep = values >= min_score
    rows = np.repeat(np.arange(scores.shape[0]), k).reshape(scores.shape[0], k)[keep]
    return rows + row_offset, idx[keep]


def generate_candidates(rep):
    """Candidate (left_index, right_index) pairs.

    Only records of the same country are compared. Signals combined (union):
    name character n-grams (typos, prefixes, abbreviations), IDF-weighted name
    tokens (rare-token blocking), space-insensitive name n-grams, address
    n-grams (street numbers + street names) and a combined name/address score.
    """
    left_codes, right_codes = rep.country_codes
    found = []
    for start in range(0, rep.n_left, config.CHUNK_SIZE):
        stop = min(start + config.CHUNK_SIZE, rep.n_left)
        block = slice(start, stop)
        mismatch = left_codes[block, None] != right_codes[None, :]

        def score(pair):
            dense = (pair[0][block] @ pair[1].T).toarray()
            dense[mismatch] = -1.0
            return dense

        name_char = score(rep.name_char)
        addr_char = score(rep.addr_char)
        signals = [
            (name_char, config.TOPK_NAME_CHAR, config.MIN_NAME_SIM),
            (score(rep.name_word), config.TOPK_NAME_WORD, config.MIN_NAME_SIM),
            (score(rep.name_squash), config.TOPK_NAME_SQUASH, config.MIN_NAME_SIM * 2),
            (addr_char, config.TOPK_ADDRESS, config.MIN_ADDRESS_SIM),
            (0.6 * name_char + 0.4 * addr_char, config.TOPK_COMBINED, config.MIN_COMBINED_SIM),
        ]
        for matrix, k, minimum in signals:
            rows, cols = _top_k_pairs(matrix, k, minimum, start)
            found.append(rows.astype(np.int64) * rep.n_right + cols)
    keys = np.unique(np.concatenate(found)) if found else np.empty(0, np.int64)
    return keys // rep.n_right, keys % rep.n_right


# ---------------------------------------------------------------------------
# PAIR FEATURES
# ---------------------------------------------------------------------------
def _row_dot(left_matrix, right_matrix, i, j, chunk=200_000):
    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        e = s + chunk
        out[s:e] = np.asarray(left_matrix[i[s:e]].multiply(right_matrix[j[s:e]]).sum(axis=1)).ravel()
    return out


def _counts(pair, i, j):
    """(intersection, size_left, size_right) of binary token sets for each pair."""
    inter = _row_dot(pair[0], pair[1], i, j)
    size_left = np.asarray(pair[0].sum(axis=1)).ravel()[i]
    size_right = np.asarray(pair[1].sum(axis=1)).ravel()[j]
    return inter, size_left, size_right


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros(len(a), dtype=np.float32), where=b > 0)


def _prefix_ratio(tokens_a, tokens_b):
    """Share of the smaller token set found in the other set exactly or as a prefix
    (handles truncated words / abbreviations such as 'intl' ~ 'international')."""
    a, b = set(tokens_a), set(tokens_b)
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    hits = 0
    for token in small:
        if token in large or (len(token) >= 3 and any(
                len(o) >= 3 and (o.startswith(token) or token.startswith(o)) for o in large)):
            hits += 1
    return hits / len(small)


def compute_pair_features(rep, i, j, natural_mask=None):
    """Feature DataFrame (columns = FEATURE_COLUMNS) for pairs (i[k], j[k]).

    ``natural_mask`` marks pairs produced by blocking; pairs injected for training
    (true matches missed by blocking) get their rank/gap features computed against
    the natural candidate list only, so training matches prediction conditions.
    """
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    if natural_mask is None:
        natural_mask = np.ones(len(i), dtype=bool)
    f = {}

    # --- name ---
    inter, size_a, size_b = _counts(rep.name_bin, i, j)
    f["name_word_jaccard"] = _safe_div(inter, size_a + size_b - inter)
    f["name_shared_tokens"] = inter
    f["name_overlap_ratio"] = _safe_div(inter, np.minimum(size_a, size_b))
    c_inter, c_a, c_b = _counts(rep.core_bin, i, j)
    f["name_core_jaccard"] = _safe_div(c_inter, c_a + c_b - c_inter)
    f["name_char_cosine"] = _row_dot(rep.name_char[0], rep.name_char[1], i, j)
    f["name_squash_cosine"] = _row_dot(rep.name_squash[0], rep.name_squash[1], i, j)
    f["name_word_cosine"] = _row_dot(rep.name_word[0], rep.name_word[1], i, j)
    f["name_exact"] = (rep.name_full[0][i] == rep.name_full[1][j]).astype(np.float32)
    f["name_core_exact"] = ((c_inter == c_a) & (c_inter == c_b) & (c_a > 0)).astype(np.float32)
    la, lb = rep.name_len[0][i], rep.name_len[1][j]
    f["name_length_ratio"] = _safe_div(np.minimum(la, lb), np.maximum(la, lb))
    first_l = np.array([t[0] if t else "" for t in rep.core_tokens[0]], dtype=object)[i]
    first_r = np.array([t[0] if t else "" for t in rep.core_tokens[1]], dtype=object)[j]
    f["name_first_token_equal"] = ((first_l == first_r) & (first_l != "")).astype(np.float32)
    f["name_prefix_ratio"] = np.fromiter(
        (_prefix_ratio(rep.core_tokens[0][a], rep.core_tokens[1][b]) for a, b in zip(i, j)),
        dtype=np.float32, count=len(i))

    # --- address ---
    a_inter, a_a, a_b = _counts(rep.addr_bin, i, j)
    f["addr_word_jaccard"] = _safe_div(a_inter, a_a + a_b - a_inter)
    f["addr_shared_tokens"] = a_inter
    f["addr_overlap_ratio"] = _safe_div(a_inter, np.minimum(a_a, a_b))
    f["addr_char_cosine"] = _row_dot(rep.addr_char[0], rep.addr_char[1], i, j)
    f["addr_word_cosine"] = _row_dot(rep.addr_word[0], rep.addr_word[1], i, j)
    n_inter, n_a, n_b = _counts(rep.num_bin, i, j)
    f["addr_numeric_shared"] = n_inter
    f["addr_numeric_jaccard"] = _safe_div(n_inter, n_a + n_b - n_inter)
    xa, xb = rep.addr_len[0][i], rep.addr_len[1][j]
    f["addr_length_ratio"] = _safe_div(np.minimum(xa, xb), np.maximum(xa, xb))
    f["addr_missing"] = ((xa == 0) | (xb == 0)).astype(np.float32)
    f["addr_exact"] = ((a_inter == a_a) & (a_inter == a_b) & (a_a > 0)).astype(np.float32)

    # --- country / source ---
    same = (rep.country_codes[0][i] == rep.country_codes[1][j]).astype(np.float32)
    f["same_country"] = same
    f["country_mismatch"] = 1.0 - same
    f["is_source3"] = rep.is_source3[j]

    # --- combined ---
    # when an address is missing the score falls back to the name alone
    f["combined_score"] = np.where(f["addr_missing"] > 0, f["name_char_cosine"],
                                   0.6 * f["name_char_cosine"] + 0.4 * f["addr_char_cosine"]).astype(np.float32)
    f["name_x_addr"] = f["name_char_cosine"] * f["addr_char_cosine"]
    f["shared_numeric_all"] = f["addr_numeric_shared"]

    features = pd.DataFrame(f)
    _add_candidate_list_features(features, i, natural_mask)
    return features[FEATURE_COLUMNS]


def _add_candidate_list_features(features, i, natural_mask):
    """Rank / gap / size of the Source 1 candidate list, based on natural candidates."""
    score = features["combined_score"].to_numpy()
    nat = pd.DataFrame({"i": i[natural_mask], "s": score[natural_mask]})
    grouped = nat.groupby("i")["s"]
    best = grouped.max()
    count = grouped.size()
    rank = np.zeros(len(i), dtype=np.float32)
    rank[natural_mask] = grouped.rank(method="min", ascending=False).to_numpy()
    if (~natural_mask).any():
        ascending = {k: np.sort(v.to_numpy()) for k, v in grouped}
        for pos in np.flatnonzero(~natural_mask):
            arr = ascending.get(i[pos])
            rank[pos] = 1 + (0 if arr is None else len(arr) - np.searchsorted(arr, score[pos], side="right"))
    best_score = pd.Series(i).map(best).fillna(pd.Series(score)).to_numpy(np.float32)
    features["cand_rank"] = rank
    features["cand_best_score"] = best_score
    features["cand_gap_to_best"] = best_score - score
    features["cand_count"] = pd.Series(i).map(count).fillna(0).to_numpy(np.float32)
