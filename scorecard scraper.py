#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 16:20:59 2026

@author: daltonstout
"""

import cv2
import numpy as np
import pandas as pd
import pytesseract
from pathlib import Path

# -----------------------------
# CONFIG YOU MUST SET ONCE
# -----------------------------
IMAGE_PATH = "/Users/daltonstout/Documents/project with willis perry/scorecard pictures/scorecard.jpeg"

# Coordinates for the rectangle containing the score GRID only:
# (x, y, width, height) in pixels.
# Tip: use the helper at the bottom to click/drag and get these numbers.
GRID_ROI = (150, 320, 1700, 600)

# Define your grid layout:
# rows = number of players (or lines) you want to read
# cols = number of holes (typically 18) OR however many score columns exist
N_ROWS = 4
N_COLS = 18

# Optional: label columns/rows
HOLE_LABELS = [f"H{i}" for i in range(1, N_COLS + 1)]
PLAYER_LABELS = [f"Player {i}" for i in range(1, N_ROWS + 1)]

# If Tesseract isn't found automatically, set the path:
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


def preprocess_cell(cell_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess a single cell image to improve handwritten digit OCR.
    """
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

    # Increase contrast a bit
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold often works better on uneven lighting
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 7
    )

    # Morphological cleanup: connect strokes, remove specks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Add a little padding border so digits aren't touching edges
    th = cv2.copyMakeBorder(th, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0)

    return th


def ocr_digit(cell_bin: np.ndarray) -> str:
    """
    OCR a single cell containing (ideally) 1–2 digits (e.g., 4, 10).
    """
    # Tesseract config tuned for short numeric text
    config = (
        "--oem 1 "
        "--psm 10 "                # treat the image as a single character
        "-c tessedit_char_whitelist=0123456789"
    )

    # Sometimes two-digit scores show up; psm 10 can fail.
    # We'll try psm 10 first, then fallback to psm 7 (single text line).
    text = pytesseract.image_to_string(cell_bin, config=config).strip()

    if not text:
        config2 = (
            "--oem 1 "
            "--psm 7 "               # single line
            "-c tessedit_char_whitelist=0123456789"
        )
        text = pytesseract.image_to_string(cell_bin, config=config2).strip()

    return text


def clean_score(s: str):
    """
    Convert OCR output to an integer score if possible, else NaN.
    """
    if not s:
        return np.nan

    # Keep digits only
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        return np.nan

    # Golf scores are typically within a reasonable range
    try:
        val = int(digits)
        if 1 <= val <= 20:
            return val
        return np.nan
    except ValueError:
        return np.nan


def split_grid_into_cells(grid_bgr: np.ndarray, n_rows: int, n_cols: int):
    """
    Split grid ROI into evenly spaced cells.
    This assumes the score boxes are uniformly spaced.
    """
    h, w = grid_bgr.shape[:2]
    cell_h = h / n_rows
    cell_w = w / n_cols

    cells = []
    for r in range(n_rows):
        row_cells = []
        for c in range(n_cols):
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)

            cell = grid_bgr[y1:y2, x1:x2].copy()

            # Optional: crop a bit inside to avoid grid lines
            pad_x = max(1, int(0.08 * (x2 - x1)))
            pad_y = max(1, int(0.10 * (y2 - y1)))
            cell = cell[pad_y:-pad_y, pad_x:-pad_x]

            row_cells.append(cell)
        cells.append(row_cells)

    return cells


def extract_scores(image_path: str) -> pd.DataFrame:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    x, y, w, h = GRID_ROI
    grid = img[y:y + h, x:x + w]

    cells = split_grid_into_cells(grid, N_ROWS, N_COLS)

    data = []
    for r in range(N_ROWS):
        row_scores = []
        for c in range(N_COLS):
            cell_bin = preprocess_cell(cells[r][c])
            raw = ocr_digit(cell_bin)
            score = clean_score(raw)
            row_scores.append(score)
        data.append(row_scores)

    df = pd.DataFrame(data, columns=HOLE_LABELS)
    df.insert(0, "player", PLAYER_LABELS)
    df["total"] = df[HOLE_LABELS].sum(axis=1, skipna=True)
    df["avg"] = df[HOLE_LABELS].mean(axis=1, skipna=True)
    return df


def make_quick_visual(df: pd.DataFrame, out_png="score_totals.png"):
    """
    Simple example visual: total score per player.
    """
    import matplotlib.pyplot as plt

    plot_df = df[["player", "total"]].copy()
    plot_df = plot_df.sort_values("total", ascending=True)

    plt.figure()
    plt.bar(plot_df["player"], plot_df["total"])
    plt.title("Total Score by Player")
    plt.xlabel("Player")
    plt.ylabel("Total")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


if __name__ == "__main__":
    df = extract_scores(IMAGE_PATH)
    print(df)

    out_csv = Path("scores_extracted.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.resolve()}")

    make_quick_visual(df)
    print("Saved: score_totals.png")

    # If OCR results are bad, first fix GRID_ROI and/or photo quality.
    # Then consider switching to a handwriting OCR model (see notes below).
