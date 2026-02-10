#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 13:09:45 2026

@author: daltonstout
"""

from __future__ import annotations

# scorecard_ocr.py
#
# Extract handwritten hole scores from a golf scorecard photo.
#
# Dependencies:
#   pip install opencv-python pytesseract numpy
#
# System dependency (Tesseract OCR engine):
#   - macOS (brew): brew install tesseract
#   - Ubuntu: sudo apt-get install tesseract-ocr
#
# Usage (CLI):
#   python scorecard_ocr.py /path/to/scorecard.jpeg --debug_dir ./debug

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract


# ----------------------------
# Data structures (app-friendly)
# ----------------------------

@dataclass
class CellOCR:
    hole: int
    score: Optional[int]
    raw_text: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class ScorecardResult:
    holes: Dict[int, Optional[int]]          # {1: 4, 2: 5, ...}
    cells: List[CellOCR]                     # detailed per-cell debug
    warped_size: Tuple[int, int]             # (w, h)
    confidence_notes: List[str]              # warnings / hints


# ----------------------------
# Core API function for your app
# ----------------------------

def extract_scores_from_image(
    image_path: str,
    *,
    tesseract_cmd: Optional[str] = None,
    debug_dir: Optional[str] = None,
) -> ScorecardResult:
    """
    Main entry point:
      - Loads image
      - Finds scorecard boundary -> perspective warp
      - Finds score grid -> cell boxes
      - OCR handwritten scores from the most "ink-heavy" row of score cells

    Returns a ScorecardResult (JSON-serializable if you convert dataclasses).
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    notes: List[str] = []

    # 1) Rotate to landscape if needed (scorecards are usually wider than tall after warp)
    #    We'll still do a proper warp first; but pre-rotation can help contour finding.
    img = _maybe_rotate_upright(img)

    # 2) Find the scorecard contour and warp to a top-down view
    warped = _warp_largest_document(img)
    if warped is None:
        notes.append("Could not confidently find scorecard boundary; using original image.")
        warped = img.copy()

    warped = _maybe_rotate_landscape(warped)
    wW, wH = warped.shape[1], warped.shape[0]

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "01_warped.jpg"), warped)

    # 3) Detect table grid and extract cell boxes
    cell_boxes = _detect_grid_cells(warped, debug_dir=debug_dir)
    if not cell_boxes:
        notes.append("Could not detect grid cells (table lines). Try better lighting / closer photo.")
        return ScorecardResult(
            holes={i: None for i in range(1, 19)},
            cells=[],
            warped_size=(wW, wH),
            confidence_notes=notes,
        )

    # 4) Group cells into rows; pick the row most likely to be the handwritten-score row
    rows = _group_cells_into_rows(cell_boxes)
    score_row = _pick_most_likely_score_row(warped, rows)

    if score_row is None:
        notes.append("Could not identify the handwritten score row reliably.")
        return ScorecardResult(
            holes={i: None for i in range(1, 19)},
            cells=[],
            warped_size=(wW, wH),
            confidence_notes=notes,
        )

    # 5) Sort left-to-right, OCR each cell, map to holes.
    #    This scorecard design shows holes 1–9 and 10–18 in different sections.
    #    This script will output up to 18 scores if it finds ~18 cells in the chosen row.
    score_row_sorted = sorted(score_row, key=lambda b: b[0])

    # If we got more than 18, keep the best 18 by "inkiness" (heuristic)
    if len(score_row_sorted) > 18:
        score_row_sorted = _keep_inkiest_n_cells(warped, score_row_sorted, n=18)

    # If we got fewer than 9, it likely picked a wrong row
    if len(score_row_sorted) < 9:
        notes.append(f"Only found {len(score_row_sorted)} score-like cells; might be the wrong row.")

    ocr_cells: List[CellOCR] = []
    holes_out: Dict[int, Optional[int]] = {i: None for i in range(1, 19)}

    # Decide hole numbering:
    # - If 18 cells: assume holes 1..18
    # - If 9 cells: assume this is either front or back; we label 1..9
    # You can extend this later by detecting BOTH score rows (front + back).
    if len(score_row_sorted) >= 18:
        hole_numbers = list(range(1, 19))
    else:
        hole_numbers = list(range(1, 1 + len(score_row_sorted)))

    for hole, (x, y, w, h) in zip(hole_numbers, score_row_sorted):
        raw, parsed = _ocr_handwritten_digit(warped, (x, y, w, h))
        ocr_cells.append(CellOCR(hole=hole, score=parsed, raw_text=raw, bbox=(x, y, w, h)))
        holes_out[hole] = parsed

    if debug_dir:
        vis = warped.copy()
        for c in ocr_cells:
            x, y, w, h = c.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{c.hole}:{c.score if c.score is not None else '?'}",
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(os.path.join(debug_dir, "02_detected_scores.jpg"), vis)

    return ScorecardResult(
        holes=holes_out,
        cells=ocr_cells,
        warped_size=(wW, wH),
        confidence_notes=notes,
    )


# ----------------------------
# OCR + Vision helpers
# ----------------------------

def _maybe_rotate_upright(img: np.ndarray) -> np.ndarray:
    """
    Quick heuristic: if photo is very tall, try rotating to help contour detection.
    """
    h, w = img.shape[:2]
    if h > w * 1.2:
        # keep as-is; many photos are portrait
        return img
    return img


def _maybe_rotate_landscape(img: np.ndarray) -> np.ndarray:
    """
    After warping, scorecard is usually landscape.
    """
    h, w = img.shape[:2]
    if h > w:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def _warp_largest_document(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the largest 4-point contour (document boundary) and perspective-warp.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:10]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * img.shape[0] * img.shape[1]:
            pts = approx.reshape(4, 2).astype(np.float32)
            return _four_point_warp(img, pts)

    return None


def _four_point_warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def _order_points(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def _detect_grid_cells(warped: np.ndarray, debug_dir: Optional[str] = None) -> List[Tuple[int, int, int, int]]:
    """
    Detect rectangular grid cells by extracting horizontal/vertical lines and finding intersections.
    Returns list of cell bounding boxes (x, y, w, h).
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Boost contrast a bit
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Binarize (table lines pop)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "10_thresh.jpg"), thr)

    # Extract horizontal and vertical lines via morphology
    h, w = thr.shape[:2]
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 40), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 40)))

    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
    vert = cv2.morphologyEx(thr, cv2.MORPH_OPEN, vert_kernel, iterations=2)

    grid = cv2.bitwise_or(horiz, vert)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "11_horiz.jpg"), horiz)
        cv2.imwrite(os.path.join(debug_dir, "12_vert.jpg"), vert)
        cv2.imwrite(os.path.join(debug_dir, "13_grid.jpg"), grid)

    # Find cell-like rectangles from grid
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh

        # Heuristics: ignore tiny noise and huge outer border boxes
        if area < 400:
            continue
        if area > (w * h) * 0.25:
            continue
        if bw < 12 or bh < 12:
            continue

        boxes.append((x, y, bw, bh))

    # De-duplicate overlapping boxes
    boxes = _non_max_suppression_boxes(boxes, overlapThresh=0.4)

    if debug_dir:
        vis = warped.copy()
        for (x, y, bw, bh) in boxes:
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "14_cells.jpg"), vis)

    return boxes


def _non_max_suppression_boxes(
    boxes: List[Tuple[int, int, int, int]],
    overlapThresh: float = 0.3,
) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []

    b = np.array(boxes, dtype=np.float32)
    x1 = b[:, 0]
    y1 = b[:, 1]
    x2 = b[:, 0] + b[:, 2]
    y2 = b[:, 1] + b[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(areas)  # small to large

    pick = []
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        idxs = idxs[:-1]

        if len(idxs) == 0:
            break

        xx1 = np.maximum(x1[last], x1[idxs])
        yy1 = np.maximum(y1[last], y1[idxs])
        xx2 = np.minimum(x2[last], x2[idxs])
        yy2 = np.minimum(y2[last], y2[idxs])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[idxs]

        idxs = idxs[overlap < overlapThresh]

    return [boxes[i] for i in pick]


def _group_cells_into_rows(boxes: List[Tuple[int, int, int, int]]) -> List[List[Tuple[int, int, int, int]]]:
    """
    Cluster boxes by y-center into rows.
    """
    if not boxes:
        return []

    boxes_sorted = sorted(boxes, key=lambda b: (b[1] + b[3] / 2, b[0]))
    rows: List[List[Tuple[int, int, int, int]]] = []

    current: List[Tuple[int, int, int, int]] = []
    current_y: Optional[float] = None

    for b in boxes_sorted:
        x, y, w, h = b
        cy = y + h / 2

        if current_y is None:
            current_y = cy
            current = [b]
            continue

        # y distance threshold relative to typical height
        avg_h = np.mean([bb[3] for bb in current])
        if abs(cy - current_y) <= max(10, avg_h * 0.6):
            current.append(b)
            current_y = (current_y * (len(current) - 1) + cy) / len(current)
        else:
            rows.append(current)
            current = [b]
            current_y = cy

    if current:
        rows.append(current)

    # Keep rows that look like score rows (many similar-sized cells)
    cleaned: List[List[Tuple[int, int, int, int]]] = []
    for r in rows:
        if len(r) < 6:
            continue
        cleaned.append(r)

    return cleaned


def _ink_density(gray_or_bgr: np.ndarray) -> float:
    """
    Return fraction of dark pixels (a proxy for handwriting amount).
    """
    if len(gray_or_bgr.shape) == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr

    # Threshold for dark pixels
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(np.mean(bw > 0))


def _pick_most_likely_score_row(
    warped: np.ndarray,
    rows: List[List[Tuple[int, int, int, int]]],
) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Pick the row whose cells contain the most ink (handwriting),
    with a mild preference for rows with ~9 or ~18 cells.
    """
    if not rows:
        return None

    scored = []
    for r in rows:
        # score by median ink density across cells
        densities = []
        for (x, y, w, h) in r:
            cell = warped[y:y+h, x:x+w]
            densities.append(_ink_density(cell))
        med = float(np.median(densities)) if densities else 0.0

        size_bonus = 0.0
        if 8 <= len(r) <= 10:
            size_bonus = 0.10
        elif 16 <= len(r) <= 20:
            size_bonus = 0.15

        scored.append((med + size_bonus, r))

    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best_row = scored[0]

    # Guardrail: if it has almost no ink, it's probably not the handwritten row
    if best_score < 0.03:
        return None
    return best_row


def _keep_inkiest_n_cells(
    warped: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    n: int,
) -> List[Tuple[int, int, int, int]]:
    scored = []
    for b in boxes:
        x, y, w, h = b
        cell = warped[y:y+h, x:x+w]
        scored.append((_ink_density(cell), b))
    scored.sort(key=lambda t: t[0], reverse=True)
    best = [b for _, b in scored[:n]]
    return sorted(best, key=lambda b: b[0])


def _ocr_handwritten_digit(warped: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, Optional[int]]:
    """
    OCR a single cell, expecting a 1–2 digit score (0–9, 10–20).
    """
    x, y, w, h = bbox
    cell = warped[y:y+h, x:x+w]

    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Try a few preprocess variants. Handwriting quality and lighting can vary a lot,
    # so a single threshold strategy often misses digits entirely.
    preprocess_variants = [
        cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8
        ),
        cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8
        ),
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    ]

    tesseract_configs = [
        "--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789",
        "--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789",
    ]

    best_raw = ""
    best_num: Optional[int] = None

    for variant in preprocess_variants:
        # Light denoising: remove tiny specks while keeping most stroke content.
        cleaned_variant = cv2.morphologyEx(
            variant, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1
        )
        enlarged = cv2.resize(cleaned_variant, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        for config in tesseract_configs:
            raw = pytesseract.image_to_string(enlarged, config=config).strip()
            digits = "".join(ch for ch in raw if ch.isdigit())
            if not digits:
                continue

            try:
                value = int(digits)
            except ValueError:
                continue

            if 1 <= value <= 20:
                return raw, value

            if best_num is None:
                best_raw = raw
                best_num = value

    return best_raw, best_num


# ----------------------------
# CLI wrapper
# ----------------------------

def _to_jsonable(result: ScorecardResult) -> dict:
    return {
        "holes": result.holes,
        "warped_size": list(result.warped_size),
        "confidence_notes": result.confidence_notes,
        "cells": [
            {
                "hole": c.hole,
                "score": c.score,
                "raw_text": c.raw_text,
                "bbox": list(c.bbox),
            }
            for c in result.cells
        ],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to scorecard photo")
    parser.add_argument("--debug_dir", default=None, help="Write intermediate images for tuning")
    parser.add_argument("--tesseract_cmd", default=None, help="Full path to tesseract binary if needed")
    parser.add_argument("--json_out", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    res = extract_scores_from_image(
        args.image_path,
        tesseract_cmd=args.tesseract_cmd,
        debug_dir=args.debug_dir,
    )

    payload = _to_jsonable(res)
    print(json.dumps(payload, indent=2))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
