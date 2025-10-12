"""
Page Alignment Algorithm for Document Comparison

Handles cases where pages are inserted, deleted, or reordered between document versions.
Uses content-based matching (text + geometry) to align pages intelligently.

Example:
    Old Doc: [A, B, C, D, E]  (5 pages)
    New Doc: [A, B, X, C, D, E, F]  (7 pages - inserted X after B, added F at end)

    Result alignment:
        Old 1 → New 1  (A matches A)
        Old 2 → New 2  (B matches B)
        None  → New 3  (X is new)
        Old 3 → New 4  (C matches C)
        Old 4 → New 5  (D matches D)
        Old 5 → New 6  (E matches E)
        None  → New 7  (F is new)
"""

from __future__ import annotations
import sqlite3
from typing import List, Tuple, Optional, Dict
import hashlib


# -------------------------
# Page Fingerprinting
# -------------------------

def _compute_page_fingerprint(conn: sqlite3.Connection, doc_id: str, page: int) -> dict:
    """
    Compute a content-based fingerprint for a page.

    Returns dict with:
        - text_hash: Hash of all text content (order-preserved)
        - text_sample: First 500 chars for quick comparison
        - geom_count: Number of geometry elements
        - geom_hash: Hash of geometry bounding boxes
        - text_count: Number of text runs
    """
    # Get all text
    text_rows = conn.execute(
        "SELECT text FROM text_rows WHERE doc_id=? AND page_number=? ORDER BY rowid",
        (doc_id, page)
    ).fetchall()

    full_text = "\n".join(row[0] for row in text_rows)
    text_hash = hashlib.sha256(full_text.encode('utf-8')).hexdigest()[:16]
    text_sample = full_text[:500] if full_text else ""

    # Get geometry stats
    geom_rows = conn.execute(
        "SELECT x0, y0, x1, y1 FROM geometry WHERE doc_id=? AND page_number=? ORDER BY rowid",
        (doc_id, page)
    ).fetchall()

    geom_bbox_str = ";".join(f"{x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}" for x0, y0, x1, y1 in geom_rows)
    geom_hash = hashlib.sha256(geom_bbox_str.encode('utf-8')).hexdigest()[:16]

    return {
        "page": page,
        "text_hash": text_hash,
        "text_sample": text_sample,
        "text_count": len(text_rows),
        "geom_hash": geom_hash,
        "geom_count": len(geom_rows),
    }


def _page_similarity(fp1: dict, fp2: dict) -> float:
    """
    Calculate similarity score between two page fingerprints.

    Returns float between 0.0 (completely different) and 1.0 (identical).

    Scoring:
        - Exact text match: +0.6
        - Text sample similarity: +0.2 (based on overlap)
        - Geometry match: +0.2
    """
    score = 0.0

    # Exact text hash match (60% weight)
    if fp1["text_hash"] == fp2["text_hash"] and fp1["text_hash"]:
        score += 0.6
    else:
        # Partial text match based on sample overlap
        s1 = set(fp1["text_sample"].split())
        s2 = set(fp2["text_sample"].split())
        if s1 and s2:
            overlap = len(s1 & s2) / max(len(s1), len(s2))
            score += 0.2 * overlap

    # Geometry match (20% weight)
    if fp1["geom_hash"] == fp2["geom_hash"] and fp1["geom_hash"]:
        score += 0.2
    elif fp1["geom_count"] > 0 and fp2["geom_count"] > 0:
        # Similar geometry count
        count_ratio = min(fp1["geom_count"], fp2["geom_count"]) / max(fp1["geom_count"], fp2["geom_count"])
        score += 0.1 * count_ratio

    # Empty page handling
    if fp1["text_count"] == 0 and fp2["text_count"] == 0 and \
       fp1["geom_count"] == 0 and fp2["geom_count"] == 0:
        return 1.0  # Both empty = perfect match

    return score


# -------------------------
# Alignment Algorithms
# -------------------------

def align_pages_greedy(
    conn: sqlite3.Connection,
    old_id: str,
    new_id: str,
    similarity_threshold: float = 0.5
) -> List[Tuple[Optional[int], Optional[int], float]]:
    """
    Align pages using a greedy algorithm.

    Returns list of (old_page, new_page, similarity_score) tuples.
    - (N, M, score): Page N in old aligns with page M in new
    - (N, None, 0.0): Page N was deleted
    - (None, M, 0.0): Page M was inserted

    Args:
        similarity_threshold: Minimum score to consider pages matched (0.0-1.0)
    """
    # Get page counts
    old_count = conn.execute(
        "SELECT page_count FROM documents WHERE doc_id=?", (old_id,)
    ).fetchone()[0]

    new_count = conn.execute(
        "SELECT page_count FROM documents WHERE doc_id=?", (new_id,)
    ).fetchone()[0]

    # Compute fingerprints for all pages
    old_fps = [_compute_page_fingerprint(conn, old_id, p) for p in range(1, old_count + 1)]
    new_fps = [_compute_page_fingerprint(conn, new_id, p) for p in range(1, new_count + 1)]

    # Greedy matching: for each old page, find best new page
    alignments: List[Tuple[Optional[int], Optional[int], float]] = []
    used_new_pages: set[int] = set()

    for old_fp in old_fps:
        best_new_idx = None
        best_score = similarity_threshold

        for new_idx, new_fp in enumerate(new_fps):
            if (new_idx + 1) in used_new_pages:
                continue

            score = _page_similarity(old_fp, new_fp)
            if score > best_score:
                best_score = score
                best_new_idx = new_idx

        if best_new_idx is not None:
            # Found a match
            new_page = best_new_idx + 1
            used_new_pages.add(new_page)
            alignments.append((old_fp["page"], new_page, best_score))
        else:
            # Page was deleted
            alignments.append((old_fp["page"], None, 0.0))

    # Add unmatched new pages (insertions)
    for new_idx, new_fp in enumerate(new_fps):
        new_page = new_idx + 1
        if new_page not in used_new_pages:
            alignments.append((None, new_page, 0.0))

    # Sort by the page that exists (old if present, else new)
    alignments.sort(key=lambda x: x[0] if x[0] is not None else (1000000 + x[1]))

    return alignments


def align_pages_dynamic(
    conn: sqlite3.Connection,
    old_id: str,
    new_id: str,
    similarity_threshold: float = 0.5
) -> List[Tuple[Optional[int], Optional[int], float]]:
    """
    Align pages using dynamic programming (optimal alignment).

    Similar to sequence alignment (Needleman-Wunsch algorithm).
    Finds the best overall alignment considering:
        - Matches: pages with high similarity
        - Insertions: new pages added
        - Deletions: old pages removed

    This is more accurate than greedy but slower for large documents.
    """
    # Get page counts
    old_count = conn.execute(
        "SELECT page_count FROM documents WHERE doc_id=?", (old_id,)
    ).fetchone()[0]

    new_count = conn.execute(
        "SELECT page_count FROM documents WHERE doc_id=?", (new_id,)
    ).fetchone()[0]

    # Compute fingerprints
    old_fps = [_compute_page_fingerprint(conn, old_id, p) for p in range(1, old_count + 1)]
    new_fps = [_compute_page_fingerprint(conn, new_id, p) for p in range(1, new_count + 1)]

    # Build similarity matrix
    sim_matrix = []
    for old_fp in old_fps:
        row = [_page_similarity(old_fp, new_fp) for new_fp in new_fps]
        sim_matrix.append(row)

    # Dynamic programming: find optimal alignment
    # dp[i][j] = best score aligning old[0..i-1] with new[0..j-1]
    dp = [[0.0] * (new_count + 1) for _ in range(old_count + 1)]
    backtrack = [[None] * (new_count + 1) for _ in range(old_count + 1)]

    # Gap penalties (for insertions/deletions)
    GAP_PENALTY = -0.1

    # Initialize first row (all insertions)
    for j in range(1, new_count + 1):
        dp[0][j] = j * GAP_PENALTY
        backtrack[0][j] = "insert"

    # Initialize first column (all deletions)
    for i in range(1, old_count + 1):
        dp[i][0] = i * GAP_PENALTY
        backtrack[i][0] = "delete"

    # Fill DP table
    for i in range(1, old_count + 1):
        for j in range(1, new_count + 1):
            # Option 1: Match/mismatch
            match_score = dp[i-1][j-1] + sim_matrix[i-1][j-1]

            # Option 2: Delete from old
            delete_score = dp[i-1][j] + GAP_PENALTY

            # Option 3: Insert in new
            insert_score = dp[i][j-1] + GAP_PENALTY

            # Take best option
            if match_score >= delete_score and match_score >= insert_score:
                dp[i][j] = match_score
                backtrack[i][j] = "match"
            elif delete_score >= insert_score:
                dp[i][j] = delete_score
                backtrack[i][j] = "delete"
            else:
                dp[i][j] = insert_score
                backtrack[i][j] = "insert"

    # Backtrack to build alignment
    alignments: List[Tuple[Optional[int], Optional[int], float]] = []
    i, j = old_count, new_count

    while i > 0 or j > 0:
        move = backtrack[i][j]

        if move == "match":
            score = sim_matrix[i-1][j-1]
            if score >= similarity_threshold:
                alignments.append((i, j, score))
            else:
                # Below threshold: treat as delete + insert
                alignments.append((i, None, 0.0))
                alignments.append((None, j, 0.0))
            i -= 1
            j -= 1
        elif move == "delete":
            alignments.append((i, None, 0.0))
            i -= 1
        elif move == "insert":
            alignments.append((None, j, 0.0))
            j -= 1
        else:
            # Edge case: both at 0
            break

    alignments.reverse()
    return alignments


# -------------------------
# High-level API
# -------------------------

def align_pages(
    conn: sqlite3.Connection,
    old_id: str,
    new_id: str,
    method: str = "dynamic",
    similarity_threshold: float = 0.5
) -> List[Tuple[Optional[int], Optional[int], float]]:
    """
    Align pages between two documents using content-based matching.

    Args:
        conn: Database connection
        old_id: Old document ID
        new_id: New document ID
        method: "greedy" (faster) or "dynamic" (optimal, default)
        similarity_threshold: Minimum similarity to consider pages matched (0.5 = 50%)

    Returns:
        List of (old_page, new_page, similarity_score):
            - (N, M, score): Page N in old matches page M in new
            - (N, None, 0.0): Page N was deleted
            - (None, M, 0.0): Page M was inserted

    Example:
        alignments = align_pages(conn, "doc_old", "doc_new")
        for old_pg, new_pg, score in alignments:
            if old_pg and new_pg:
                print(f"Page {old_pg} → {new_pg} (similarity: {score:.2f})")
            elif old_pg:
                print(f"Page {old_pg} was DELETED")
            else:
                print(f"Page {new_pg} was INSERTED")
    """
    if method == "greedy":
        return align_pages_greedy(conn, old_id, new_id, similarity_threshold)
    elif method == "dynamic":
        return align_pages_dynamic(conn, old_id, new_id, similarity_threshold)
    else:
        raise ValueError(f"Unknown alignment method: {method}. Use 'greedy' or 'dynamic'.")


def get_alignment_summary(alignments: List[Tuple[Optional[int], Optional[int], float]]) -> dict:
    """
    Generate a human-readable summary of page alignment results.
    """
    matched = sum(1 for old, new, _ in alignments if old and new)
    deleted = sum(1 for old, new, _ in alignments if old and not new)
    inserted = sum(1 for old, new, _ in alignments if not old and new)

    avg_similarity = 0.0
    if matched > 0:
        avg_similarity = sum(score for old, new, score in alignments if old and new) / matched

    return {
        "total_alignments": len(alignments),
        "matched_pages": matched,
        "deleted_pages": deleted,
        "inserted_pages": inserted,
        "average_similarity": avg_similarity,
    }


__all__ = [
    "align_pages",
    "align_pages_greedy",
    "align_pages_dynamic",
    "get_alignment_summary",
]
