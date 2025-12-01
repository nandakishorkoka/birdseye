"""Utilities for visualizing GPS points from camera detections."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd
from gmplot import GoogleMapPlotter

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT = BASE_DIR / "images" / "camera_1_map.html"
DEFAULT_MARKER_COLOR = "#111111"
DEFAULT_LIGHT_GRAY = "#777777"


def _hex_to_byte(component: str) -> int:
    return int(component.lstrip("#")[:2], 16)


def _gray_hex(value: int) -> str:
    clamped = max(0, min(255, value))
    return f"#{clamped:02x}{clamped:02x}{clamped:02x}"


def grayscale_palette(
    count: int,
    dark_hex: str = DEFAULT_MARKER_COLOR,
    light_hex: str = DEFAULT_LIGHT_GRAY,
) -> list[str]:
    if count <= 0:
        raise ValueError("Count must be a positive integer")

    start = _hex_to_byte(dark_hex)
    end = _hex_to_byte(light_hex)
    if count == 1 or start == end:
        return [_gray_hex(start)]

    step = (end - start) / (count - 1)
    return [_gray_hex(round(start + (step * i))) for i in range(count)]


def load_lat_lon(csv_path: Path) -> Tuple[list[float], list[float]]:
    """Read latitude and longitude columns from the provided CSV file."""
    df = pd.read_csv(csv_path)
    try:
        lat_series = df["latitude"]
        lon_series = df["longitude"]
    except KeyError as exc:
        raise ValueError(
            f"{csv_path.name} must contain 'latitude' and 'longitude' columns"
        ) from exc

    return lat_series.astype(float).tolist(), lon_series.astype(float).tolist()


def collect_csv_files(directory: Path) -> Sequence[Path]:
    """Return sorted CSV files from the provided directory."""
    if not directory.exists():
        raise ValueError(f"{directory} does not exist")
    all_files = sorted(directory.glob("*.csv"))
    csv_files = [path for path in all_files if path.is_file()]
    if not csv_files:
        raise ValueError(f"No CSV files found in {directory}")
    return csv_files


def load_all_lat_lon(directory: Path) -> list[Tuple[list[float], list[float], str]]:
    """Aggregate latitude/longitude pairs across every CSV in `directory`."""
    csv_files = collect_csv_files(directory)
    palette = grayscale_palette(len(csv_files))
    batches: list[Tuple[list[float], list[float], str]] = []
    for csv_path, color in zip(csv_files, palette):
        file_lats, file_lons = load_lat_lon(csv_path)
        batches.append((file_lats, file_lons, color))
    return batches


def draw_squares(
    plotter: GoogleMapPlotter,
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    half_edge: float = 0.0002,
    color: str = DEFAULT_MARKER_COLOR,
    edge_width: float = 1,
    face_alpha: float = 0.6,
) -> None:
    """Overlay a small square centered at each GPS location."""
    for lat, lon in zip(latitudes, longitudes):
        square_lats = [lat - half_edge, lat - half_edge, lat + half_edge, lat + half_edge]
        square_lons = [lon - half_edge, lon + half_edge, lon + half_edge, lon - half_edge]
        plotter.polygon(
            square_lats,
            square_lons,
            face_color=color,
            edge_color=color,
            edge_width=edge_width,
            face_alpha=face_alpha,
        )


def pixels_to_degrees(zoom: int) -> float:
    """Estimate longitude degrees covered by a single pixel at the given zoom."""
    return 360.0 / (256 * (2 ** zoom))


def build_map(
    batches: Sequence[Tuple[list[float], list[float], str]],
    output_path: Path,
    zoom: int = 17,
    half_edge: float | None = None,
    marker_pixels: float = 3.0,
    api_key: str | None = None,
) -> Path:
    """Create a Google map with GPS squares and write the HTML output."""
    if not batches:
        raise ValueError("At least one CSV batch is required.")

    expanded_lats = [lat for latitudes, _, _ in batches for lat in latitudes]
    expanded_lons = [lon for _, longitudes, _ in batches for lon in longitudes]

    if not expanded_lats or not expanded_lons:
        raise ValueError("At least one latitude/longitude pair is required.")

    center_lat = sum(expanded_lats) / len(expanded_lats)
    center_lon = sum(expanded_lons) / len(expanded_lons)
    if half_edge is None:
        degree_per_pixel = pixels_to_degrees(zoom)
        half_edge = (marker_pixels / 2.0) * degree_per_pixel
    kwargs = {"apikey": api_key} if api_key else {}
    plotter = GoogleMapPlotter(center_lat, center_lon, zoom, **kwargs)
    for latitudes, longitudes, color in batches:
        draw_squares(
            plotter,
            latitudes,
            longitudes,
            half_edge=half_edge,
            color=color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.draw(str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render camera detections as squares on a Google map."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing CSV files with latitude/longitude columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination HTML file for the generated map",
    )
    parser.add_argument(
        "--square-half-edge",
        type=float,
        default=None,
        help=(
            "Optional half the length of each square's edge (in degrees). "
            "If omitted, the script uses 3 pixels at the chosen zoom."
        ),
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=17,
        help="Initial Google Maps zoom level",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "Google Maps API key (optional). "
            "If omitted, gmplot falls back to the public embedding."
        ),
    )
    parser.add_argument(
        "--marker-pixels",
        type=float,
        default=3.0,
        help="Size of the square markers in pixels when --square-half-edge is not set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batches = load_all_lat_lon(args.csv_dir)
    output_path = build_map(
        batches,
        args.output,
        zoom=args.zoom,
        half_edge=args.square_half_edge,
        marker_pixels=args.marker_pixels,
        api_key=args.api_key,
    )
    print(f"Map saved to {output_path}")


if __name__ == "__main__":
    main()
