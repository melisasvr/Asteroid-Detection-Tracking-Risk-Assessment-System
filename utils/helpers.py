# utils/helpers.py
"""
Shared I/O and preprocessing helpers.

If you've used Rasterio before, think of these as the FITS equivalents:
  - load_fits_image  ≈  rasterio.open()  +  .read()
  - normalize_image  ≈  rio.plot.show normalization
  - sigma_clip_background ≈ masking nodata / cloud pixels
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FITS I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_fits_image(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a FITS image and return (pixel_array, header_dict).

    FITS is astronomy's equivalent of GeoTIFF:
      - The primary HDU (Header Data Unit) holds pixel data + metadata.
      - Header keywords carry WCS (world coordinate system), exposure time,
        instrument info — same idea as GeoTIFF tags.

    Returns
    -------
    data   : 2-D float32 array, shape (rows, cols)
    header : dict of header keywords
    """
    try:
        from astropy.io import fits
    except ImportError:
        raise ImportError("astropy is required: pip install astropy")

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {filepath}")

    with fits.open(filepath) as hdul:
        # hdul is a list of HDUs; index 0 is usually the primary image.
        # Some instruments store science data in extension 1 — handle both.
        sci_hdu = hdul[0]
        if sci_hdu.data is None and len(hdul) > 1:
            sci_hdu = hdul[1]

        data   = sci_hdu.data.astype(np.float32)
        header = dict(sci_hdu.header)

    # FITS can be 3-D (multi-channel or time-stack); collapse to 2-D
    if data.ndim == 3:
        data = data[0]          # take first frame
    elif data.ndim > 3:
        data = data[0, 0]

    logger.info(f"Loaded FITS {filepath}  shape={data.shape}  dtype={data.dtype}")
    return data, header


def save_fits_image(data: np.ndarray, filepath: str,
                    header: Optional[Dict] = None) -> None:
    """Write a numpy array back to FITS (useful for saving detection masks)."""
    from astropy.io import fits
    hdu = fits.PrimaryHDU(data)
    if header:
        for k, v in header.items():
            try:
                hdu.header[k] = v
            except Exception:
                pass
    hdu.writeto(filepath, overwrite=True)
    logger.info(f"Saved FITS → {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def sigma_clip_background(image: np.ndarray,
                           sigma: float = 3.0,
                           iters: int = 5) -> Tuple[float, float]:
    """
    Iterative σ-clipping to estimate background mean and std.

    Why σ-clipping instead of plain mean/std?
    Stars, cosmic rays, and asteroids are OUTLIERS in a telescope image —
    they'd inflate your standard deviation and raise your detection threshold.
    Clipping removes those bright pixels iteratively until the stats converge
    on a stable background estimate.

    Identical concept to masking cloud pixels before computing NDVI stats.

    Returns
    -------
    (background_mean, background_std)
    """
    data = image.flatten().copy()
    for _ in range(iters):
        mu  = np.mean(data)
        std = np.std(data)
        mask = np.abs(data - mu) < sigma * std
        if mask.sum() == len(data):   # converged — nothing more to clip
            break
        data = data[mask]
    return float(np.mean(data)), float(np.std(data))


def normalize_image(image: np.ndarray,
                    low_pct: float = 1.0,
                    high_pct: float = 99.0) -> np.ndarray:
    """
    Percentile-stretch normalize to [0, 1].

    Same as 'linear stretch' in QGIS/ENVI — you clip extreme values
    so faint objects become visible without saturating on bright stars.
    """
    lo  = np.percentile(image, low_pct)
    hi  = np.percentile(image, high_pct)
    out = np.clip((image - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return out.astype(np.float32)


def subtract_background(image: np.ndarray,
                         box_size: int = 64) -> np.ndarray:
    """
    Simple median-filter background subtraction (mesh-based).

    Divides image into boxes, computes median in each box, then
    bilinearly interpolates back to full resolution. This handles
    large-scale gradients from sky glow / detector patterns —
    the same reason you remove trends before analysing satellite reflectance.
    """
    from scipy.ndimage import zoom, median_filter
    smoothed = median_filter(image, size=box_size)
    return (image - smoothed).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Results I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Serialize pipeline results to JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types → native Python so JSON doesn't choke
    def _convert(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(filepath, "w") as f:
        json.dump(results, f, default=_convert, indent=2)
    logger.info(f"Results saved → {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    with open(filepath) as f:
        return json.load(f)