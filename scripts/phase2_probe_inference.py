"""Probe CPU inference of one FaceAnalysis variant in a fresh process.

Run once per round and per variant so cold init is never shared. `all-modules`
rebuilds the pre-change runtime; the first run of it writes the reference
gallery that later runs are compared against.
"""

import argparse
import csv
import hashlib
import json
import resource
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.domain.errors import DomainError
from app.ml import InsightFaceEmbedder
from scripts.sample_images import face_image_bytes, sample_image_path

VARIANTS = {
    "all-modules": None,
    "det-rec": ["detection", "recognition"],
}
# Vùng ảnh nhóm t1.jpg chứa đúng hai khuôn mặt.
_TWO_FACE_CROP = (slice(120, 440), slice(240, 600))
_FIELDS = [
    "git_sha",
    "variant",
    "modules",
    "cold_init_ms",
    "hot_median_ms",
    "hot_p95_ms",
    "rss_peak_mb",
    "rss_after_hot_mb",
    "gallery_size",
    "max_abs_diff",
    "min_self_cosine",
    "top1_matches",
    "decisions_match",
    "parallel_max_abs_diff",
    "errors",
]


def _jpeg(image: np.ndarray) -> bytes:
    encoded, payload = cv2.imencode(".jpg", image)
    if not encoded:
        raise RuntimeError("Cannot encode a probe image")
    return payload.tobytes()


def _gallery_images(analysis: FaceAnalysis) -> list[bytes]:
    """Cut every face of the bundled group picture into its own one-face image."""
    image = cv2.imread(str(sample_image_path()))
    height, width = image.shape[:2]
    crops = []
    for x1, y1, x2, y2 in sorted(face.bbox.astype(int).tolist() for face in analysis.get(image)):
        margin_x, margin_y = (x2 - x1) // 2, (y2 - y1) // 2
        crops.append(
            _jpeg(
                image[
                    max(y1 - margin_y, 0) : min(y2 + margin_y, height),
                    max(x1 - margin_x, 0) : min(x2 + margin_x, width),
                ]
            )
        )
    return crops


def _error_code(embedder: InsightFaceEmbedder, payload: bytes) -> str:
    try:
        embedder.embed(payload)
    except DomainError as error:
        return type(error).__name__
    return "ok"


def _rss_mb() -> float:
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * resource.getpagesize() / 1024**2


def _percentile_95(samples: list[float]) -> float:
    return statistics.quantiles(samples, n=100, method="inclusive")[94]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    started = time.perf_counter()
    analysis = FaceAnalysis(
        name="buffalo_s",
        allowed_modules=VARIANTS[args.variant],
        providers=["CPUExecutionProvider"],
    )
    analysis.prepare(ctx_id=0, det_size=(640, 640))
    cold_init_ms = (time.perf_counter() - started) * 1000
    embedder = InsightFaceEmbedder(analysis=analysis)

    face = face_image_bytes()
    for _ in range(args.warmup):
        embedder.embed(face)
    hot_ms = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        embedder.embed(face)
        hot_ms.append((time.perf_counter() - started) * 1000)
    rss_after_hot_mb = _rss_mb()

    images = _gallery_images(analysis)
    digests = [hashlib.sha256(image).hexdigest() for image in images]
    vectors = np.stack([embedder.embed(image) for image in images])
    if not args.reference.exists():
        if args.variant != "all-modules":
            parser.error("create the reference with --variant all-modules first")
        np.savez(args.reference, vectors=vectors, digests=np.array(digests))
    with np.load(args.reference) as reference:
        if reference["digests"].tolist() != digests:
            raise RuntimeError("Probe images differ from the reference run")
        profiles = reference["vectors"]
    # Profile do runtime gốc tạo; mỗi ảnh phải khớp chính profile của nó.
    scores = vectors @ profiles.T
    own = np.arange(len(images))

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        parallel = np.stack(list(pool.map(embedder.embed, [face] * (args.parallel * 4))))

    sample = cv2.imread(str(sample_image_path()))
    result = {
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "variant": args.variant,
        "modules": " ".join(sorted(analysis.models)),
        "cold_init_ms": round(cold_init_ms, 1),
        "hot_median_ms": round(statistics.median(hot_ms), 2),
        "hot_p95_ms": round(_percentile_95(hot_ms), 2),
        "rss_peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "rss_after_hot_mb": round(rss_after_hot_mb, 1),
        "gallery_size": len(images),
        "max_abs_diff": float(np.abs(vectors - profiles).max()),
        "min_self_cosine": float(scores[own, own].min()),
        "top1_matches": int((scores.argmax(axis=1) == own).sum()),
        "decisions_match": bool(
            np.array_equal(scores >= args.threshold, profiles @ profiles.T >= args.threshold)
        ),
        "parallel_max_abs_diff": float(np.abs(parallel - embedder.embed(face)).max()),
        "errors": json.dumps(
            {
                "empty": _error_code(embedder, b""),
                "corrupt": _error_code(embedder, b"not-an-image"),
                "no_face": _error_code(embedder, _jpeg(np.zeros((128, 128, 3), dtype=np.uint8))),
                "two_faces": _error_code(embedder, _jpeg(sample[_TWO_FACE_CROP])),
            },
            sort_keys=True,
        ),
    }

    is_new = not args.csv.exists()
    with args.csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
