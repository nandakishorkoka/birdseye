"""Utility to compute and pickle a homography from image to UTM coordinates."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute homography matrix from image points to UTM coordinates."
    )
    parser.add_argument("csv_path", help="Path to the CSV file with point correspondences.")
    return parser.parse_args()


def load_correspondences(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 4:
        raise ValueError("CSV must have at least four columns (x, y, utm_x, utm_y).")
    image_pts = data[:, :2].astype(np.float64)
    utm_pts = data[:, 2:4].astype(np.float64)
    return image_pts, utm_pts


def compute_homography(image_pts: np.ndarray, utm_pts: np.ndarray) -> np.ndarray:
    matrix, status = cv2.findHomography(image_pts, utm_pts)
    if matrix is None:
        raise RuntimeError("Failed to compute homography matrix with provided points.")
    return matrix


def save_matrix(matrix: np.ndarray, csv_path: Path) -> Path:
    pickle_path = csv_path.with_suffix(".pkl")
    with open(pickle_path, "wb") as fh:
        pickle.dump(matrix, fh)
    return pickle_path


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).resolve()
    image_pts, utm_pts = load_correspondences(csv_path)
    homography = compute_homography(image_pts, utm_pts)
    output_path = save_matrix(homography, csv_path)
    print(f"Saved homography matrix to {output_path}")


if __name__ == "__main__":
    main()

