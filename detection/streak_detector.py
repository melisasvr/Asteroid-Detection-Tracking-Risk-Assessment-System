# detection/streak_detector.py
"""
Asteroid Streak Detector
========================
Detects linear moving-object trails in telescope images using classical
computer vision. This is the "brains" of the detection pipeline.

ALGORITHM OVERVIEW
------------------
1. Preprocess  → background subtract, σ-clip threshold → binary mask
2. Morphology  → dilate mask to connect broken streaks
3. Label       → connected-components (like rasterio's vectorize, but pixels not polygons)
4. Geometry    → fit ellipses to each blob; filter by aspect ratio (streaks are elongated)
5. Hough       → verify surviving candidates with Probabilistic Hough Line Transform
6. Filter      → apply physical constraints (min length, max width, angle consistency)
7. Output      → list of DetectedStreak objects with positions + confidence score

WHY THIS APPROACH?
------------------
The classical pipeline is FAST (sub-second per image), interpretable (you can
visualise every step), and robust to noise. It's the same approach used in
real survey pipelines (LSST's moving-object pipeline uses similar ideas).
The CNN (Module 3) then re-scores candidates — combining CV's speed with ML's
pattern recognition.

Think of it as: CV does the coarse search, ML does the verification.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structure for a detected streak
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectedStreak:
    """Represents one detected linear feature in the image."""
    # Endpoints (pixel coordinates)
    x1: float
    y1: float
    x2: float
    y2: float
    # Geometric properties
    length_px:   float   # trail length in pixels
    width_px:    float   # trail FWHM (cross-section width)
    angle_deg:   float   # orientation (degrees from horizontal, -90 to 90)
    # Photometric
    peak_flux:   float   # brightest pixel value along streak
    total_flux:  float   # integrated flux in the aperture
    snr:         float   # signal-to-noise ratio
    # Detection confidence
    confidence:  float   # combined CV score ∈ [0, 1]
    # Optional: ML re-score (filled in by the CNN classifier later)
    ml_score:    float = -1.0
    # Bounding box (for CNN cutout extraction)
    bbox: Tuple[int,int,int,int] = field(default_factory=lambda: (0,0,0,0))

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def aspect_ratio(self) -> float:
        return self.length_px / max(self.width_px, 0.1)

    @property
    def is_candidate(self) -> bool:
        """High-confidence candidate: strong CV + ML agreement."""
        return self.confidence > 0.4 and (self.ml_score < 0 or self.ml_score > 0.5)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'x1': round(self.x1, 2), 'y1': round(self.y1, 2),
            'x2': round(self.x2, 2), 'y2': round(self.y2, 2),
            'length_px': round(self.length_px, 2),
            'width_px':  round(self.width_px, 2),
            'angle_deg': round(self.angle_deg, 2),
            'snr':       round(self.snr, 2),
            'confidence':round(self.confidence, 3),
            'ml_score':  round(self.ml_score, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main detector class
# ─────────────────────────────────────────────────────────────────────────────

class StreakDetector:
    """
    Full classical computer-vision streak detector.

    Parameters
    ----------
    sigma_threshold : float
        Detection threshold in units of background σ. Default 5.0.
        Lower = more sensitive but more false positives.
    min_length_px : int
        Minimum trail length to consider. Shorter blobs are stars / noise.
    max_width_px : float
        Maximum cross-section width. Wider blobs are likely stars or galaxies.
    min_aspect_ratio : float
        length / width floor. Below this, the blob is too round to be a streak.
    hough_threshold : int
        Hough accumulator threshold. Lower = more lines found.
    """

    def __init__(self,
                 sigma_threshold:  float = 5.0,
                 min_length_px:    int   = 10,
                 max_width_px:     float = 8.0,
                 min_aspect_ratio: float = 3.0,
                 hough_threshold:  int   = 20):

        self.sigma_threshold  = sigma_threshold
        self.min_length_px    = min_length_px
        self.max_width_px     = max_width_px
        self.min_aspect_ratio = min_aspect_ratio
        self.hough_threshold  = hough_threshold

        # Saved intermediate images (for dashboard visualisation)
        self.debug_images: Dict[str, np.ndarray] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> List[DetectedStreak]:
        """
        Run the full detection pipeline on a 2-D float32 image.
        Returns list of DetectedStreak objects, sorted by confidence (desc).
        """
        logger.info(f"Starting streak detection on image {image.shape}")

        # Step 1 — Background estimation and thresholding
        bg_sub, binary_mask = self._preprocess(image)

        # Step 2 — Morphological cleanup
        cleaned_mask = self._morphological_filter(binary_mask)

        # Step 3 — Connected-component labelling
        blobs = self._label_components(cleaned_mask)
        logger.info(f"  Found {len(blobs)} raw blobs after labelling")

        # Step 4 — Geometric filtering (aspect ratio, size)
        candidates = self._filter_by_geometry(blobs, cleaned_mask)
        logger.info(f"  {len(candidates)} candidates after geometry filter")

        # Step 5 — Hough refinement (sub-pixel endpoint estimation)
        streaks = self._hough_refine(candidates, binary_mask, bg_sub)
        logger.info(f"  {len(streaks)} streaks after Hough refinement")

        # Step 6 — Photometry and confidence scoring
        streaks = self._score_streaks(streaks, bg_sub)

        # Sort by confidence descending
        streaks.sort(key=lambda s: s.confidence, reverse=True)
        logger.info(f"  Returning {len(streaks)} final detections")
        return streaks

    def detect_from_file(self, fits_path: str) -> List[DetectedStreak]:
        """Load a FITS file and run detection."""
        from utils.helpers import load_fits_image, subtract_background
        image, _ = load_fits_image(fits_path)
        return self.detect(image)

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Background subtract + threshold → binary detection mask.

        Steps:
        a) Median background subtraction (removes large-scale sky gradients)
        b) σ-clip statistics on background-subtracted image
        c) Binary threshold at N×σ above background mean

        Analogy to your satellite work:
        This is exactly like computing a "change detection" mask —
        you're isolating pixels that are "significantly different"
        from the baseline background.
        """
        from scipy.ndimage import median_filter

        # Median filter with large kernel removes all point sources,
        # leaving a smooth background model
        bg_model  = median_filter(image.astype(np.float64), size=31)
        bg_sub    = (image - bg_model).astype(np.float32)

        # σ-clip on background-subtracted image
        flat  = bg_sub.flatten()
        mu    = np.median(flat)                    # median more robust than mean
        sigma = np.median(np.abs(flat - mu)) * 1.4826  # MAD → σ (robust estimator)

        threshold = mu + self.sigma_threshold * sigma
        binary    = (bg_sub > threshold).astype(np.uint8) * 255

        self.debug_images['background_subtracted'] = bg_sub
        self.debug_images['binary_mask'] = binary
        self._bg_mu    = float(mu)
        self._bg_sigma = float(sigma)

        return bg_sub, binary

    # ── Step 2: Morphological filtering ───────────────────────────────────────

    def _morphological_filter(self, binary: np.ndarray) -> np.ndarray:
        """
        Dilate the binary mask to bridge gaps in faint streaks.

        Real asteroid streaks are often NOT continuously bright — the PSF
        cross-section may dip below threshold between sub-exposures.
        A small dilation (3×1 horizontal + 1×3 vertical kernel) connects
        nearby bright pixels belonging to the same trail without merging
        separated objects.

        Then we close with a slightly larger kernel to fill interior holes.
        """
        # Structuring element: 3×3 cross — connects 4-connected neighbours
        kernel_cross  = cv2.getStructuringElement(cv2.MORPH_CROSS,  (3, 3))
        kernel_rect   = cv2.getStructuringElement(cv2.MORPH_RECT,   (5, 5))

        dilated = cv2.dilate(binary, kernel_cross, iterations=2)
        closed  = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_rect)

        self.debug_images['morphology'] = closed
        return closed

    # ── Step 3: Connected-component labelling ─────────────────────────────────

    def _label_components(self, mask: np.ndarray) -> List[Dict]:
        """
        Find connected blobs and extract their bounding boxes + stats.

        cv2.connectedComponentsWithStats returns:
          - num_labels : total labels (including background = 0)
          - labels     : same shape as mask, each pixel has its label int
          - stats      : (N, 5) array: [x, y, w, h, area] per label
          - centroids  : (N, 2) array: centroid [cx, cy] per label

        Background label = 0 is always skipped.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        blobs = []
        for i in range(1, num_labels):   # skip 0 = background
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]

            if area < 5:   # too small to be anything real
                continue

            blobs.append({
                'label': i,
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': area,
                'cx': cx, 'cy': cy,
                'bbox': (x, y, x + w, y + h),
            })

        return blobs

    # ── Step 4: Geometric filtering ───────────────────────────────────────────

    def _filter_by_geometry(self, blobs: List[Dict],
                             mask: np.ndarray) -> List[Dict]:
        """
        Keep only blobs that are geometrically streak-like.

        Key insight: a streak is ELONGATED (high aspect ratio) and LINEAR.
        Stars are circular (aspect ≈ 1). Cosmic ray clusters can be irregular.

        We fit a minimum-area bounding rectangle (rotated) to each blob's
        pixel footprint — this gives us:
          - oriented_length : longer dimension
          - oriented_width  : shorter dimension
          - angle           : orientation angle

        This is much more informative than the axis-aligned bbox, because a
        45-degree diagonal streak has w≈h in the axis-aligned box but clear
        elongation in the rotated box.
        """
        candidates = []

        for blob in blobs:
            x0, y0, x1, y1 = blob['bbox']
            # Extract pixel coordinates of this blob
            region = mask[y0:y1, x0:x1]
            pts = np.column_stack(np.where(region > 0))  # (N, 2) [row, col]

            if len(pts) < 5:
                continue

            # Fit minimum-area rotated rectangle
            pts_xy = pts[:, [1, 0]].astype(np.float32)  # → (col, row) = (x, y)
            rect   = cv2.minAreaRect(pts_xy)             # returns ((cx,cy),(w,h),angle)
            (rcx, rcy), (rw, rh), angle = rect

            oriented_len   = max(rw, rh)
            oriented_width = min(rw, rh)

            if oriented_len < self.min_length_px:
                continue
            if oriented_width > self.max_width_px:
                continue
            aspect = oriented_len / max(oriented_width, 0.1)
            if aspect < self.min_aspect_ratio:
                continue

            # Convert rect angle to standard degrees-from-horizontal convention
            if rw < rh:
                angle = angle + 90
            angle = angle % 180 - 90    # → [-90, 90]

            blob.update({
                'oriented_len':   oriented_len,
                'oriented_width': oriented_width,
                'aspect':         aspect,
                'angle':          angle,
                'rect':           rect,
                # Shift centroid back to full-image coordinates
                'rcx': rcx + x0,
                'rcy': rcy + y0,
            })
            candidates.append(blob)

        return candidates

    # ── Step 5: Hough line refinement ─────────────────────────────────────────

    def _hough_refine(self, candidates: List[Dict],
                      binary: np.ndarray,
                      bg_sub: np.ndarray) -> List[DetectedStreak]:
        """
        Probabilistic Hough Transform to get precise sub-pixel endpoints.

        WHY HOUGH?
        ----------
        The bounding-box approach gives us an approximate position, but the
        endpoints are tied to the bbox corners. Hough finds the actual line
        that maximises the number of collinear bright pixels → better astrometry.

        Probabilistic Hough (HoughLinesP) is faster than standard Hough for
        extracting segments (it doesn't need to scan all rho/theta values).
        Parameters:
          rho        : accumulator resolution (pixels)
          theta      : angular resolution (radians)
          threshold  : minimum votes for a line
          minLineLen : minimum segment length to return
          maxLineGap : maximum gap to bridge in a line segment
        """
        streaks = []

        for cand in candidates:
            x0, y0, x1, y1 = cand['bbox']
            # Small padding to handle streaks at bbox edge
            pad = 5
            ry0 = max(0, y0 - pad)
            ry1 = min(binary.shape[0], y1 + pad)
            rx0 = max(0, x0 - pad)
            rx1 = min(binary.shape[1], x1 + pad)

            roi = binary[ry0:ry1, rx0:rx1]

            lines = cv2.HoughLinesP(
                roi,
                rho        = 1,
                theta      = np.pi / 180,
                threshold  = self.hough_threshold,
                minLineLength = max(5, int(self.min_length_px * 0.8)),
                maxLineGap = 5,
            )

            if lines is None:
                # Hough found no lines — fall back to bbox geometry
                lx1 = cand['rcx'] - cand['oriented_len'] / 2 * np.cos(np.radians(cand['angle']))
                ly1 = cand['rcy'] - cand['oriented_len'] / 2 * np.sin(np.radians(cand['angle']))
                lx2 = cand['rcx'] + cand['oriented_len'] / 2 * np.cos(np.radians(cand['angle']))
                ly2 = cand['rcy'] + cand['oriented_len'] / 2 * np.sin(np.radians(cand['angle']))
            else:
                # Pick the longest Hough segment
                best = max(lines, key=lambda l: np.hypot(l[0][2]-l[0][0], l[0][3]-l[0][1]))
                hx1, hy1, hx2, hy2 = best[0]
                # Convert ROI coordinates back to full-image coordinates
                lx1, ly1 = hx1 + rx0, hy1 + ry0
                lx2, ly2 = hx2 + rx0, hy2 + ry0

            length = np.hypot(lx2 - lx1, ly2 - ly1)
            angle  = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))

            streak = DetectedStreak(
                x1=float(lx1), y1=float(ly1),
                x2=float(lx2), y2=float(ly2),
                length_px=float(length),
                width_px=float(cand['oriented_width']),
                angle_deg=float(angle),
                peak_flux=0.0, total_flux=0.0, snr=0.0,  # filled in next step
                confidence=0.0,
                bbox=(int(rx0), int(ry0), int(rx1), int(ry1)),
            )
            streaks.append(streak)

        return streaks

    # ── Step 6: Photometry and confidence ─────────────────────────────────────

    def _score_streaks(self, streaks: List[DetectedStreak],
                       bg_sub: np.ndarray) -> List[DetectedStreak]:
        """
        Measure photometry along each streak and compute a confidence score.

        CONFIDENCE SCORE COMPONENTS
        ---------------------------
        1. SNR score       : clipped(SNR / 15) → high SNR → high score
        2. Aspect score    : tanh(aspect / 5)  → longer, narrower → higher
        3. Length score    : clipped(length / 50) → penalise very short trails
        4. Combined        : weighted average of the three

        This is a heuristic score — not a trained classifier. The CNN module
        will replace this with a learned probability.
        """
        rows, cols = bg_sub.shape
        scored = []

        for s in streaks:
            # Sample pixels along the streak (10 evenly-spaced samples)
            n_samples = max(10, int(s.length_px))
            xs = np.linspace(s.x1, s.x2, n_samples)
            ys = np.linspace(s.y1, s.y2, n_samples)

            # Bilinear sampling (like rasterio's sample() method)
            xi  = np.clip(xs.astype(int), 0, cols - 1)
            yi  = np.clip(ys.astype(int), 0, rows - 1)
            vals = bg_sub[yi, xi]

            peak_flux  = float(np.max(vals))
            total_flux = float(np.sum(np.maximum(vals, 0)))
            snr        = peak_flux / max(self._bg_sigma, 0.01)

            # Confidence score
            snr_score    = float(np.clip(snr / 15.0, 0, 1))
            aspect_score = float(np.tanh(s.aspect_ratio / 5.0))
            length_score = float(np.clip(s.length_px / 50.0, 0, 1))
            confidence   = 0.5 * snr_score + 0.3 * aspect_score + 0.2 * length_score

            s.peak_flux  = peak_flux
            s.total_flux = total_flux
            s.snr        = snr
            s.confidence = confidence
            scored.append(s)

        return scored


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers (compare detections to ground truth)
# ─────────────────────────────────────────────────────────────────────────────

def match_detections_to_truth(detections: List[DetectedStreak],
                               truths,
                               tolerance_px: float = 15.0) -> Dict:
    """
    Simple nearest-neighbour matching between detections and ground truth.
    Returns dict with TP, FP, FN counts and per-match details.

    Matching criterion: midpoint distance < tolerance_px.
    """
    matched_truth = set()
    matched_det   = set()
    matches       = []

    for di, det in enumerate(detections):
        dx, dy = det.midpoint
        best_dist = np.inf
        best_ti   = -1

        for ti, truth in enumerate(truths):
            if ti in matched_truth:
                continue
            tx = (truth.x_start + truth.x_end) / 2
            ty = (truth.y_start + truth.y_end) / 2
            dist = np.hypot(dx - tx, dy - ty)
            if dist < best_dist:
                best_dist = dist
                best_ti   = ti

        if best_dist < tolerance_px and best_ti >= 0:
            matched_truth.add(best_ti)
            matched_det.add(di)
            matches.append({
                'detection_idx': di,
                'truth_idx':     best_ti,
                'distance_px':   round(best_dist, 2),
            })

    tp = len(matches)
    fp = len(detections) - tp
    fn = len(truths) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(precision, 3),
        'recall':    round(recall, 3),
        'f1':        round(f1, 3),
        'matches':   matches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from data.simulator import TelescopeImageSimulator

    sim   = TelescopeImageSimulator()
    image, truths = sim.generate()

    detector = StreakDetector(sigma_threshold=4.0, min_length_px=8)
    streaks  = detector.detect(image)

    print(f"\nDetections: {len(streaks)}")
    for s in streaks:
        print(f"  ({s.x1:.0f},{s.y1:.0f})→({s.x2:.0f},{s.y2:.0f})  "
              f"len={s.length_px:.0f}px  SNR={s.snr:.1f}  conf={s.confidence:.2f}")

    metrics = match_detections_to_truth(streaks, truths)
    print(f"\nEvaluation vs. ground truth:")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print(f"  Precision={metrics['precision']:.2f}  Recall={metrics['recall']:.2f}  F1={metrics['f1']:.2f}")