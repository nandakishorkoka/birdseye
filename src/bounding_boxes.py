from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from PIL import Image, ImageDraw
from ultralytics import YOLO


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect person bounding boxes using YOLO and export them to CSV."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory to save CSV outputs.",
    )
    parser.add_argument(
        "--overlay-suffix",
        type=str,
        default="_bbox",
        help="Suffix to append when saving the labeled image.",
    )
    return parser.parse_args()


def _resolve_person_class(model: YOLO) -> int:
    names: Mapping[int, str] | Sequence[str] = getattr(model, "names", {})
    if isinstance(names, Mapping):
        for idx, name in names.items():
            if isinstance(name, str) and name.lower() == "person":
                return idx
    elif isinstance(names, Sequence):
        for idx, name in enumerate(names):
            if isinstance(name, str) and name.lower() == "person":
                return idx
    return 0


def _extract_person_boxes(result, person_cls: int) -> Iterable[tuple[float, float, float, float, float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()

    try:
        coords = boxes.xyxy.tolist()
        confidences = boxes.conf.tolist()
        class_indices = boxes.cls.tolist()
    except AttributeError:
        return ()

    rows = []
    for (x1, y1, x2, y2), confidence, cls in zip(coords, confidences, class_indices):
        if int(cls) != person_cls:
            continue
        rows.append((float(x1), float(y1), float(x2), float(y2), float(confidence)))
    return rows


def _write_csv(output_path: Path, rows: Iterable[tuple[float, float, float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["foot_x", "foot_y", "confidence"])
        writer.writerows(rows)


def _overlay_boxes(
    image_path: Path,
    rows: Sequence[tuple[float, float, float, float, float]],
    suffix: str,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, _confidence in rows:
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
    overlay_path = image_path.with_name(f"{image_path.stem}{suffix}{image_path.suffix}")
    image.save(overlay_path)
    return overlay_path


def main() -> None:
    args = _parse_args()
    image_path = args.image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # model = YOLO("yolov10s.pt")
    model = YOLO("yolov8x.pt") 
    person_cls = _resolve_person_class(model)
    results = model(str(image_path))

    if not results:
        raise RuntimeError("Model did not return any results.")

    person_boxes = list(_extract_person_boxes(results[0], person_cls))
    footpoints = [
        (((x1 + x2) / 2.0), y2, confidence)
        for x1, _y1, x2, y2, confidence in person_boxes
    ]
    output_file = args.output_dir / f"{image_path.stem}.csv"
    _write_csv(output_file, footpoints)
    overlay_file = _overlay_boxes(image_path, person_boxes, args.overlay_suffix)
    print(f"Saved {len(person_boxes)} person boxes to {output_file}")
    print(f"Overlay image saved to {overlay_file}")


if __name__ == "__main__":
    main()
