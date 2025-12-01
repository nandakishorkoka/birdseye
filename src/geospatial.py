"""Transform bounding box foot points into latitude/longitude coordinates."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from pyproj import Transformer
except ImportError as exc:  # pragma: no cover - informs missing dependency
    raise RuntimeError(
        "pyproj is required to convert UTM to latitude/longitude. "
        "Run `pip install pyproj` before using src/geospatial.py."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project foot points from an image CSV into WGS84."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to a CSV produced by src/bounding_boxes.py with foot_x/foot_y.",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs",
        help="Directory containing the matching <name>.pkl homography.",
    )
    parser.add_argument(
        "--source-crs",
        type=str,
        default="EPSG:32618",
        help="Source CRS of homography output (defaults to UTM zone 18N).",
    )
    return parser.parse_args()


def load_points(
    csv_path: Path,
) -> tuple[list[dict[str, str]], list[tuple[float, float]], list[str]]:
    csv_path = csv_path.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"No CSV found at {csv_path}")
    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("CSV is empty or missing a header row.")
        rows: list[dict[str, str]] = []
        footpoints: list[tuple[float, float]] = []
        for index, row in enumerate(reader, start=1):
            try:
                foot_x = float(row["foot_x"])
                foot_y = float(row["foot_y"])
            except KeyError as exc:
                raise KeyError("CSV must include 'foot_x' and 'foot_y' columns.") from exc
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric foot coordinates on row {index}: {row}"
                ) from exc
            footpoints.append((foot_x, foot_y))
            rows.append(row)
    return rows, footpoints, list(fieldnames)


def load_homography_matrix(configs_dir: Path, csv_path: Path) -> np.ndarray:
    configs_dir = configs_dir.resolve()
    pickle_path = configs_dir / csv_path.with_suffix(".pkl").name
    if not pickle_path.exists():
        raise FileNotFoundError(f"Missing homography pickle at {pickle_path}")
    with pickle_path.open("rb") as fh:
        matrix = pickle.load(fh)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Expected a 3x3 homography matrix.")
    return matrix


def apply_homography(
    matrix: np.ndarray, points: Sequence[tuple[float, float]]
) -> Iterable[tuple[float, float]]:
    for x, y in points:
        source = np.array([x, y, 1.0], dtype=np.float64)
        target = matrix @ source
        if abs(target[2]) < 1e-12:
            raise RuntimeError("Homography produced a point at infinity.")
        yield float(target[0] / target[2]), float(target[1] / target[2])


def write_points(
    csv_path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]
) -> None:
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def project_to_lat_lon(
    coords: Sequence[tuple[float, float]], source_crs: str
) -> list[tuple[float, float]]:
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lat_lon: list[tuple[float, float]] = []
    for easting, northing in coords:
        lon, lat = transformer.transform(easting, northing)
        lat_lon.append((lat, lon))
    return lat_lon


def main() -> None:
    args = parse_args()
    rows, footpoints, original_fieldnames = load_points(args.csv_path)
    if not rows:
        print(f"No rows to process in {args.csv_path}")
        return
    matrix = load_homography_matrix(args.configs_dir, args.csv_path)
    utm_coords = list(apply_homography(matrix, footpoints))
    lat_lon_pairs = project_to_lat_lon(utm_coords, args.source_crs)
    fieldnames = list(original_fieldnames)
    for column in ("latitude", "longitude"):
        if column not in fieldnames:
            fieldnames.append(column)
    for row, (lat, lon) in zip(rows, lat_lon_pairs):
        row["latitude"] = f"{lat:.8f}"
        row["longitude"] = f"{lon:.8f}"
    write_points(args.csv_path.resolve(), rows, fieldnames)
    print(f"Appended latitude/longitude for {len(rows)} points in {args.csv_path}")


if __name__ == "__main__":
    main()

