# data/simulator.py
"""
Synthetic FITS Image Simulator
================================
Generates realistic telescope survey images containing:
  - Gaussian background noise (sky + detector readout noise)
  - Point sources (stars) following a realistic magnitude distribution
  - Asteroid streaks (linear motion across exposure time)
  - Cosmic rays (short bright linear artifacts — a real detection challenge)

WHY SIMULATE?
-------------
Real telescope FITS archives exist (SDSS, TESS, NEOWISE), but downloading
and aligning them takes time and pipeline complexity. Simulation lets you:
  1. Control ground truth → know EXACTLY where the asteroid is.
  2. Test edge cases (faint streaks, crowded star fields, short exposures).
  3. Generate unlimited training data for the CNN classifier.

The physics here is simplified but captures the key image characteristics
that make streak detection hard:
  - Background is NOT flat (vignetting, sky gradients).
  - Stars vary enormously in brightness (log-uniform flux distribution).
  - Asteroids are FAINT — SNR 3-10 in a single image.
  - Cosmic rays look like short streaks → false positive source.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AsteroidTruth:
    """Ground-truth record for one simulated asteroid streak."""
    object_id:   str
    x_start:     float   # pixel col at start of exposure
    y_start:     float   # pixel row at start of exposure
    x_end:       float   # pixel col at end of exposure
    y_end:       float   # pixel row at end of exposure
    flux:        float   # total integrated flux (ADU)
    snr:         float   # signal-to-noise ratio
    trail_length: float  # pixels
    angle_deg:   float   # trail angle (degrees from horizontal)

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x_start + self.x_end) / 2,
                (self.y_start + self.y_end) / 2)


@dataclass
class SimulationConfig:
    """All tuneable parameters in one place."""
    # Image geometry
    image_size:     Tuple[int, int] = (512, 512)   # (rows, cols)
    # Noise model
    sky_background:   float = 1000.0   # ADU — typical CCD sky level
    readout_noise:    float = 10.0     # ADU rms — detector electronics
    gain:             float = 1.5      # electrons per ADU
    # Stars
    n_stars:          int   = 200
    star_flux_min:    float = 500.0    # faintest star (ADU)
    star_flux_max:    float = 50000.0  # brightest star (ADU)
    star_fwhm:        float = 2.5      # PSF full-width half-maximum (pixels)
    # Asteroids
    n_asteroids:      int   = 3
    asteroid_flux_min: float = 300.0
    asteroid_flux_max: float = 2000.0
    speed_min_px:     float = 5.0    # apparent motion in pixels over exposure
    speed_max_px:     float = 40.0
    asteroid_fwhm:    float = 2.0    # PSF width along trail cross-section
    # Cosmic rays
    n_cosmic_rays:    int   = 15
    cr_flux_min:      float = 5000.0
    cr_flux_max:      float = 30000.0
    cr_length_min:    float = 3.0
    cr_length_max:    float = 12.0
    # Reproducibility
    random_seed:      Optional[int] = 42


# ─────────────────────────────────────────────────────────────────────────────
# Core simulator
# ─────────────────────────────────────────────────────────────────────────────

class TelescopeImageSimulator:
    """
    Generates synthetic telescope images with injected asteroid streaks.

    Usage
    -----
    >>> sim = TelescopeImageSimulator()
    >>> image, truths = sim.generate()
    >>> # image  : np.ndarray shape (512,512) float32
    >>> # truths : List[AsteroidTruth]
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.cfg = config or SimulationConfig()
        self.rng = np.random.default_rng(self.cfg.random_seed)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> Tuple[np.ndarray, List[AsteroidTruth]]:
        """
        Build a full simulated image.
        Returns (image_array, list_of_asteroid_truths).
        """
        rows, cols = self.cfg.image_size
        image = np.zeros((rows, cols), dtype=np.float32)

        # Layer 1: Background sky + detector noise
        image += self._make_background()

        # Layer 2: Static stars (PSF point sources)
        image += self._make_stars()

        # Layer 3: Asteroid streaks (the targets we want to detect)
        streak_layer, truths = self._make_asteroids()
        image += streak_layer

        # Layer 4: Cosmic rays (false positive challenge)
        image += self._make_cosmic_rays()

        # Clip to realistic ADU range (16-bit detector)
        image = np.clip(image, 0, 65535).astype(np.float32)
        return image, truths

    def generate_fits(self, output_path: str) -> List[AsteroidTruth]:
        """Generate image and save as a FITS file. Returns ground truths."""
        from astropy.io import fits

        image, truths = self.generate()

        hdr = fits.Header()
        hdr['INSTRUME'] = 'SIM_CAM'
        hdr['EXPTIME']  = 300.0        # seconds — affects streak length
        hdr['GAIN']     = self.cfg.gain
        hdr['RDNOISE']  = self.cfg.readout_noise
        hdr['NAXIS1']   = image.shape[1]
        hdr['NAXIS2']   = image.shape[0]
        hdr['COMMENT']  = 'AsteroidWatch simulated image'

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(output_path, image, hdr, overwrite=True)
        return truths

    # ── Private layer builders ────────────────────────────────────────────────

    def _make_background(self) -> np.ndarray:
        """
        Sky background = constant sky level + Poisson photon noise + Gaussian readout noise.

        Why two noise sources?
        - Poisson: counting photons is random → variance = mean (sky statistics)
        - Gaussian: detector electronics add fixed readout noise regardless of signal

        Combined in quadrature: total_noise = sqrt(sky + readout²)
        This is the textbook CCD noise model.
        """
        rows, cols = self.cfg.image_size
        sky = self.cfg.sky_background

        # Poisson noise around sky level (approximated as Gaussian for large sky counts)
        shot_noise    = self.rng.normal(0, np.sqrt(sky), (rows, cols))
        readout_noise = self.rng.normal(0, self.cfg.readout_noise, (rows, cols))

        # Add a subtle vignetting gradient (flux falls off toward edges — common in real optics)
        y, x = np.mgrid[:rows, :cols]
        cy, cx = rows / 2, cols / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        vignette = 1.0 - 0.15 * (r / r.max())**2   # 15% drop at corners

        return (sky * vignette + shot_noise + readout_noise).astype(np.float32)

    def _gaussian_psf(self, image: np.ndarray, cx: float, cy: float,
                       flux: float, fwhm: float) -> None:
        """
        Stamp a 2-D Gaussian PSF onto the image at (cx, cy) with given flux.

        FWHM → sigma conversion: sigma = FWHM / (2 * sqrt(2 * ln2)) ≈ FWHM / 2.355
        The Gaussian is integrated over pixels (not just sampled at centre) —
        this is the 'pixelated' PSF used in real photometry pipelines.
        """
        sigma = fwhm / 2.355
        radius = int(4 * sigma) + 1
        rows, cols = image.shape

        iy0 = max(0, int(cy) - radius)
        iy1 = min(rows, int(cy) + radius + 1)
        ix0 = max(0, int(cx) - radius)
        ix1 = min(cols, int(cx) + radius + 1)

        y = np.arange(iy0, iy1)
        x = np.arange(ix0, ix1)
        xx, yy = np.meshgrid(x, y)

        g = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        g_norm = g / (g.sum() + 1e-12)
        image[iy0:iy1, ix0:ix1] += (flux * g_norm).astype(np.float32)

    def _make_stars(self) -> np.ndarray:
        """
        Scatter stars across the image.
        Flux is drawn from a log-uniform distribution — brighter stars are rarer,
        which approximates the luminosity function of background field stars.
        """
        rows, cols = self.cfg.image_size
        layer = np.zeros((rows, cols), dtype=np.float32)

        log_min = np.log10(self.cfg.star_flux_min)
        log_max = np.log10(self.cfg.star_flux_max)

        for _ in range(self.cfg.n_stars):
            cx   = self.rng.uniform(0, cols)
            cy   = self.rng.uniform(0, rows)
            flux = 10 ** self.rng.uniform(log_min, log_max)
            self._gaussian_psf(layer, cx, cy, flux, self.cfg.star_fwhm)

        return layer

    def _make_streak(self, image: np.ndarray,
                      x0: float, y0: float,
                      x1: float, y1: float,
                      flux: float, fwhm: float) -> None:
        """
        Draw a linear streak by summing Gaussians along the trail.

        Physical model:
        During a long exposure, an asteroid MOVES. Its PSF is smeared into a
        line. We approximate this by placing N_sub Gaussian PSF stamps along
        the trajectory. Total flux is conserved — distributed equally among stamps.

        N_sub is chosen so stamps overlap (step ≈ 0.5 px) → smooth line.
        """
        length = np.hypot(x1 - x0, y1 - y0)
        n_sub  = max(2, int(length / 0.5))
        flux_per_sub = flux / n_sub

        for i in range(n_sub):
            t  = i / (n_sub - 1)
            cx = x0 + t * (x1 - x0)
            cy = y0 + t * (y1 - y0)
            self._gaussian_psf(image, cx, cy, flux_per_sub, fwhm)

    def _make_asteroids(self) -> Tuple[np.ndarray, List[AsteroidTruth]]:
        """
        Inject asteroid streaks. Returns (layer, ground_truths).

        Each asteroid gets a random start position, random velocity direction,
        and random (but faint) flux.
        """
        rows, cols = self.cfg.image_size
        layer  = np.zeros((rows, cols), dtype=np.float32)
        truths = []

        # Background noise for SNR calculation
        bg_std = np.sqrt(self.cfg.sky_background + self.cfg.readout_noise**2)

        for i in range(self.cfg.n_asteroids):
            # Random trail — must stay (mostly) inside the image
            x0    = self.rng.uniform(0.1 * cols, 0.9 * cols)
            y0    = self.rng.uniform(0.1 * rows, 0.9 * rows)
            speed = self.rng.uniform(self.cfg.speed_min_px, self.cfg.speed_max_px)
            angle = self.rng.uniform(0, 2 * np.pi)
            x1    = np.clip(x0 + speed * np.cos(angle), 0, cols - 1)
            y1    = np.clip(y0 + speed * np.sin(angle), 0, rows - 1)

            flux  = self.rng.uniform(self.cfg.asteroid_flux_min, self.cfg.asteroid_flux_max)
            self._make_streak(layer, x0, y0, x1, y1, flux, self.cfg.asteroid_fwhm)

            # Approximate SNR: peak flux per pixel / background noise
            trail_len  = np.hypot(x1 - x0, y1 - y0)
            peak_per_px = flux / max(1, trail_len) / (2 * np.pi * self.cfg.asteroid_fwhm**2 / 4)
            snr = peak_per_px / (bg_std + 1e-9)

            angle_deg = np.degrees(np.arctan2(y1 - y0, x1 - x0))

            truths.append(AsteroidTruth(
                object_id    = f"SIM-{i+1:04d}",
                x_start      = float(x0),
                y_start      = float(y0),
                x_end        = float(x1),
                y_end        = float(y1),
                flux         = float(flux),
                snr          = float(snr),
                trail_length = float(trail_len),
                angle_deg    = float(angle_deg),
            ))

        return layer, truths

    def _make_cosmic_rays(self) -> np.ndarray:
        """
        Cosmic rays = charged particles hitting the CCD → bright short linear artifacts.
        They're the #1 false positive source for streak detectors.
        Real pipelines use multiple dithered exposures to reject them (they don't repeat).
        For single-image detection, we rely on: they're very bright, very narrow, very short.
        """
        rows, cols = self.cfg.image_size
        layer = np.zeros((rows, cols), dtype=np.float32)

        for _ in range(self.cfg.n_cosmic_rays):
            x0    = self.rng.uniform(0, cols)
            y0    = self.rng.uniform(0, rows)
            length = self.rng.uniform(self.cfg.cr_length_min, self.cfg.cr_length_max)
            angle  = self.rng.uniform(0, 2 * np.pi)
            x1    = np.clip(x0 + length * np.cos(angle), 0, cols - 1)
            y1    = np.clip(y0 + length * np.sin(angle), 0, rows - 1)
            flux  = self.rng.uniform(self.cfg.cr_flux_min, self.cfg.cr_flux_max)

            # Cosmic rays are VERY narrow (sub-pixel width) — fwhm ≈ 0.7 px
            self._make_streak(layer, x0, y0, x1, y1, flux, fwhm=0.7)

        return layer


# ─────────────────────────────────────────────────────────────────────────────
# Batch generation (for ML training data)
# ─────────────────────────────────────────────────────────────────────────────

def generate_training_batch(n_images: int = 100,
                             output_dir: str = "data/processed/training",
                             seed: int = 0) -> List[Dict]:
    """
    Generate N labelled images for CNN training.
    Returns list of dicts: {'image_path': ..., 'truths': [...]}
    """
    import json
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []

    for i in range(n_images):
        cfg = SimulationConfig(random_seed=seed + i)
        sim = TelescopeImageSimulator(cfg)
        img_path  = str(out / f"image_{i:04d}.npy")
        truth_path = str(out / f"truth_{i:04d}.json")

        image, truths = sim.generate()
        np.save(img_path, image)

        truth_dicts = [
            {k: (v if not isinstance(v, float) else round(v, 4))
             for k, v in t.__dict__.items()}
            for t in truths
        ]
        with open(truth_path, "w") as f:
            json.dump(truth_dicts, f, indent=2)

        records.append({'image_path': img_path, 'truths': truth_dicts})

    print(f"Generated {n_images} training images in {output_dir}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.helpers import normalize_image

    sim   = TelescopeImageSimulator()
    image, truths = sim.generate()

    print(f"Image shape : {image.shape}")
    print(f"Image range : {image.min():.0f} – {image.max():.0f} ADU")
    print(f"Asteroids   : {len(truths)}")
    for t in truths:
        print(f"  {t.object_id}  trail={t.trail_length:.1f}px  SNR≈{t.snr:.1f}  angle={t.angle_deg:.1f}°")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(normalize_image(image), cmap='gray', origin='lower')

    for t in truths:
        ax.plot([t.x_start, t.x_end], [t.y_start, t.y_end],
                'r-', linewidth=2, alpha=0.7)
        ax.annotate(t.object_id, xy=t.midpoint, color='red', fontsize=7)

    ax.set_title("Simulated Telescope Image — Red lines = ground-truth asteroid trails")
    plt.tight_layout()
    plt.savefig("data/processed/simulation_preview.png", dpi=150)
    print("Preview saved → data/processed/simulation_preview.png")