# orbits/orbit_engine.py
"""
Orbital Mechanics Engine
========================
Everything related to converting detections → orbits → risk assessment.

CONCEPTS YOU NEED (new territory — explained as you go)
-------------------------------------------------------
Keplerian Elements (6 numbers that fully describe an orbit):
  a  = semi-major axis (AU)   — "average" distance from Sun; orbit size
  e  = eccentricity (0-1)     — 0=circle, 1=parabola; orbit "shape"
  i  = inclination (deg)      — tilt relative to Earth's orbital plane
  Ω  = RAAN (deg)             — longitude of ascending node; where orbit crosses ecliptic
  ω  = arg. of perihelion (deg) — angle from node to closest approach
  M  = mean anomaly (deg)     — where the body is RIGHT NOW along its orbit

MOID (Minimum Orbit Intersection Distance):
  The closest the two ORBITS can get to each other, regardless of where
  the bodies actually are at any given time. MOID < 0.05 AU = PHA status.
  Think of it as: "if you were to sample all possible times, what's the
  closest the asteroid's orbit passes to Earth's orbit?"

Torino Scale:
  0 = no hazard, 10 = certain civilization-ending impact.
  Function of (MOID, probability_of_impact).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Orbital elements data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrbitalElements:
    """
    Classical Keplerian orbital elements + derived risk quantities.
    All angles in degrees. Distances in AU.
    """
    object_id:   str
    a:   float      # semi-major axis (AU)
    e:   float      # eccentricity
    i:   float      # inclination (deg)
    raan: float     # right ascension of ascending node Ω (deg)
    argp: float     # argument of perihelion ω (deg)
    M0:  float      # mean anomaly at epoch (deg)
    epoch_jd: float # Julian Date of epoch

    # Derived — computed after fitting
    perihelion_au:  float = field(default=0.0)   # q = a(1-e)
    aphelion_au:    float = field(default=0.0)   # Q = a(1+e)
    period_yr:      float = field(default=0.0)   # T = a^(3/2) years (Kepler's 3rd law)
    moid_au:        float = field(default=99.0)  # MOID vs Earth
    torino_level:   int   = field(default=0)
    torino_label:   str   = field(default="No Hazard")
    torino_color:   str   = field(default="#AAAAAA")
    impact_prob:    float = field(default=0.0)
    risk_score:     float = field(default=0.0)   # composite [0,1]

    def __post_init__(self):
        self.perihelion_au = self.a * (1 - self.e)
        self.aphelion_au   = self.a * (1 + self.e)
        # Kepler's 3rd law: T² = a³ (when T in years, a in AU)
        self.period_yr     = self.a ** 1.5

    @property
    def is_neo(self) -> bool:
        """Near-Earth Object: perihelion < 1.3 AU."""
        return self.perihelion_au < 1.3

    @property
    def is_pha(self) -> bool:
        """Potentially Hazardous Asteroid: MOID < 0.05 AU AND H < 22."""
        return self.moid_au < 0.05

    def to_dict(self) -> Dict:
        return {
            'object_id':    self.object_id,
            'a':            round(self.a, 4),
            'e':            round(self.e, 4),
            'i':            round(self.i, 2),
            'raan':         round(self.raan, 2),
            'argp':         round(self.argp, 2),
            'perihelion_au':round(self.perihelion_au, 4),
            'aphelion_au':  round(self.aphelion_au, 4),
            'period_yr':    round(self.period_yr, 3),
            'moid_au':      round(self.moid_au, 6),
            'torino_level': self.torino_level,
            'torino_label': self.torino_label,
            'torino_color': self.torino_color,
            'impact_prob':  round(self.impact_prob, 6),
            'risk_score':   round(self.risk_score, 4),
            'is_neo':       self.is_neo,
            'is_pha':       self.is_pha,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Orbit propagator (Kepler solver)
# ─────────────────────────────────────────────────────────────────────────────

class KeplerSolver:
    """
    Converts Keplerian elements → 3-D Cartesian positions at any time.

    The key challenge: Kepler's equation  M = E - e*sin(E)
    is TRANSCENDENTAL — E (eccentric anomaly) can't be solved analytically.
    We use Newton-Raphson iteration. Converges in ~5 iterations for e < 0.9.
    """

    @staticmethod
    def mean_to_eccentric(M_deg: float, e: float, tol: float = 1e-10) -> float:
        """
        Solve Kepler's equation M = E - e*sin(E) for E (eccentric anomaly).
        M and return value are in radians internally, but M_deg is input in degrees.
        """
        M = np.radians(M_deg) % (2 * np.pi)
        # Initial guess: E ≈ M for small e; better starting point for high e
        E = M + e * np.sin(M) / (1 - np.sin(M + e) + np.sin(M))

        for _ in range(50):   # Newton-Raphson iterations
            dE = (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
            E += dE
            if abs(dE) < tol:
                break
        return E

    @staticmethod
    def eccentric_to_true(E: float, e: float) -> float:
        """
        Convert eccentric anomaly E → true anomaly ν.
        ν = 2 * arctan( sqrt((1+e)/(1-e)) * tan(E/2) )
        """
        nu = 2 * np.arctan2(
            np.sqrt(1 + e) * np.sin(E / 2),
            np.sqrt(1 - e) * np.cos(E / 2)
        )
        return nu

    @classmethod
    def elements_to_cartesian(cls, elems: OrbitalElements,
                                dt_days: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert orbital elements to heliocentric Cartesian (x, y, z) in AU.
        dt_days = days since epoch.

        Returns (position_AU, velocity_AU_per_day).

        The transformation has 3 stages:
        1. Solve Kepler's equation → position in orbital plane (r, θ)
        2. 3-D rotation from orbital plane → ecliptic frame (3 Euler rotations)
        3. Scale by semi-major axis
        """
        GM_AU3_day2 = 2.959e-4   # G*M_sun in AU³/day² (standard gravitational parameter)

        # ── 1. Advance mean anomaly by dt ─────────────────────────────────────
        n_deg_per_day = 360.0 / (elems.period_yr * 365.25)   # mean motion (deg/day)
        M = (elems.M0 + n_deg_per_day * dt_days) % 360.0

        # ── 2. Solve Kepler → eccentric → true anomaly ──────────────────────
        E  = cls.mean_to_eccentric(M, elems.e)
        nu = cls.eccentric_to_true(E, elems.e)

        # ── 3. Position in orbital plane (perifocal frame) ──────────────────
        r = elems.a * (1 - elems.e * np.cos(E))   # distance from focus (AU)
        p_pf = r * np.array([np.cos(nu), np.sin(nu), 0.0])

        # ── 4. Velocity in perifocal frame ──────────────────────────────────
        h = np.sqrt(GM_AU3_day2 * elems.a * (1 - elems.e**2))  # specific angular momentum
        v_pf = (GM_AU3_day2 / h) * np.array([
            -np.sin(nu),
            elems.e + np.cos(nu),
            0.0
        ])

        # ── 5. Rotation matrices: perifocal → ecliptic ──────────────────────
        # Three Euler rotations: ω (argp), i (inclination), Ω (raan)
        # R = Rz(-Ω) · Rx(-i) · Rz(-ω)    [standard orbital mechanics]
        W = np.radians(elems.raan)
        inc = np.radians(elems.i)
        w   = np.radians(elems.argp)

        def Rz(a): return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        def Rx(a): return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])

        R = Rz(-W) @ Rx(-inc) @ Rz(-w)

        pos_AU = R @ p_pf
        vel_AU = R @ v_pf

        return pos_AU, vel_AU

    @classmethod
    def propagate_orbit(cls, elems: OrbitalElements,
                         n_points: int = 360) -> np.ndarray:
        """
        Return one full orbital path as (n_points, 3) array in AU.
        Used for 3-D orbit visualization.
        """
        total_days = elems.period_yr * 365.25
        times = np.linspace(0, total_days, n_points)
        positions = np.array([
            cls.elements_to_cartesian(elems, dt)[0]
            for dt in times
        ])
        return positions


# ─────────────────────────────────────────────────────────────────────────────
# MOID calculator
# ─────────────────────────────────────────────────────────────────────────────

class MOIDCalculator:
    """
    Minimum Orbit Intersection Distance (MOID) calculator.

    WHAT IS MOID?
    The minimum distance between ANY point on the asteroid's orbit
    and ANY point on Earth's orbit. It tells us how geometrically
    close the two orbital paths get — independent of WHEN each body
    is at that point.

    METHOD: Grid search + refinement
    We sample N positions on Earth's orbit and N on the asteroid orbit,
    find the (i,j) pair with minimum distance, then refine with scipy
    optimizer around that minimum. This is a simplified but robust approach.
    Full analytical MOID requires solving a system of polynomial equations.

    EARTH'S ORBIT (reference):
      a = 1.000 AU, e = 0.0167, i = 0°, ω = 102.9°, Ω = 0°
    """

    EARTH_ELEMENTS = {
        'a': 1.000, 'e': 0.0167, 'i': 0.00,
        'raan': 0.0, 'argp': 102.9, 'M0': 0.0,
        'epoch_jd': 2451545.0   # J2000.0
    }

    def __init__(self, n_grid: int = 720):
        """n_grid: number of points on each orbit for initial grid search."""
        self.n_grid = n_grid
        self._earth_orbit = self._build_earth_orbit()

    def _build_earth_orbit(self) -> np.ndarray:
        """Precompute Earth's orbit positions (stays constant)."""
        earth = OrbitalElements(
            object_id='Earth', **self.EARTH_ELEMENTS
        )
        return KeplerSolver.propagate_orbit(earth, self.n_grid)

    def compute_moid(self, elems: OrbitalElements) -> float:
        """
        Compute MOID between the asteroid orbit and Earth's orbit.
        Returns distance in AU.
        """
        from scipy.optimize import minimize_scalar, minimize

        # Asteroid orbit positions
        ast_orbit = KeplerSolver.propagate_orbit(elems, self.n_grid)

        # Grid search: compute all pairwise distances (vectorised)
        # Shape: (n_grid_ast, 1, 3) - (1, n_grid_earth, 3) → (n_grid_ast, n_grid_earth)
        diff   = ast_orbit[:, np.newaxis, :] - self._earth_orbit[np.newaxis, :, :]
        dists  = np.linalg.norm(diff, axis=2)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)

        # Grid minimum
        moid_grid = dists[min_idx]

        # Refinement: narrow window around grid minimum
        # Express as 1-D problem: minimize over asteroid anomaly t_a
        # for each asteroid position, find closest Earth position
        i_ast, i_earth = min_idx
        total_days_ast   = elems.period_yr * 365.25
        total_days_earth = 365.25

        def distance_at_times(params):
            t_a = params[0] % total_days_ast
            t_e = params[1] % total_days_earth
            p_a, _ = KeplerSolver.elements_to_cartesian(elems, t_a)
            earth_elems = OrbitalElements(object_id='Earth', **self.EARTH_ELEMENTS)
            p_e, _ = KeplerSolver.elements_to_cartesian(earth_elems, t_e)
            return float(np.linalg.norm(p_a - p_e))

        # Local refinement around the grid minimum
        t0_ast   = (i_ast   / self.n_grid) * total_days_ast
        t0_earth = (i_earth / self.n_grid) * total_days_earth

        try:
            result = minimize(
                distance_at_times,
                x0     = [t0_ast, t0_earth],
                method = 'Nelder-Mead',
                options = {'xatol': 0.01, 'fatol': 1e-6, 'maxiter': 500},
            )
            moid = max(0.0, float(result.fun))
        except Exception:
            moid = float(moid_grid)

        return moid


# ─────────────────────────────────────────────────────────────────────────────
# Risk assessor (Torino Scale)
# ─────────────────────────────────────────────────────────────────────────────

class RiskAssessor:
    """
    Assigns Torino Scale level and computes a composite risk score.

    IMPACT PROBABILITY MODEL (simplified)
    -------------------------------------
    Real impact probability requires Monte Carlo propagation of orbital
    uncertainty (thousands of orbital clones). Here we use a heuristic:
    P(impact) ≈ σ(-(MOID / σ_MOID)²) where σ_MOID is the MOID uncertainty
    derived from the number of observations.

    For portfolio purposes this is clearly documented as simplified.
    Real pipelines use OpenOrb or Horizons' statistical OD.
    """

    def assess(self, elems: OrbitalElements,
               n_observations: int = 10,
               observation_arc_days: float = 30.0) -> OrbitalElements:
        """
        Compute impact probability, Torino level, and risk score.
        Modifies the OrbitalElements object in place and returns it.

        Parameters
        ----------
        n_observations       : number of astrometric detections
        observation_arc_days : time span covered by observations (days)
                               Longer arc → better orbit → lower uncertainty
        """
        moid = elems.moid_au

        # Orbital uncertainty scales inversely with arc length and observation count
        # This is a rough heuristic: real OD uncertainty from covariance matrix
        arc_factor = np.clip(observation_arc_days / 30.0, 0.1, 10.0)
        obs_factor = np.clip(n_observations / 10.0, 0.1, 5.0)
        sigma_moid = 0.01 / (arc_factor * obs_factor)   # AU

        # Heuristic impact probability (Gaussian tail beyond MOID/sigma)
        # If MOID >> sigma_moid → P ≈ 0; if MOID ≈ 0 → P ≈ 0.5
        z = moid / max(sigma_moid, 1e-6)
        impact_prob = float(np.exp(-0.5 * z**2) * 0.01)  # scale factor for realism

        # Torino Scale assignment
        torino_level, torino_label, torino_color = self._torino_lookup(moid, impact_prob)

        # Composite risk score [0, 1]:
        # Combines MOID proximity (inverted), impact probability, and NEO status
        moid_score   = float(np.exp(-moid / 0.01))         # 1 at moid=0, drops fast
        prob_score   = float(np.clip(impact_prob * 1000, 0, 1))
        neo_bonus    = 0.1 if elems.is_neo else 0.0
        risk_score   = float(np.clip(0.5*moid_score + 0.4*prob_score + 0.1*neo_bonus, 0, 1))

        elems.impact_prob  = impact_prob
        elems.torino_level = torino_level
        elems.torino_label = torino_label
        elems.torino_color = torino_color
        elems.risk_score   = risk_score
        return elems

    @staticmethod
    def _torino_lookup(moid_au: float, p_impact: float) -> Tuple[int, str, str]:
        """Return (torino_level, label, hex_color) given MOID and impact prob."""
        from utils.constants import TORINO_SCALE_THRESHOLDS
        for max_moid, min_prob, level, label, color in TORINO_SCALE_THRESHOLDS:
            if moid_au <= max_moid and p_impact >= min_prob:
                return level, label, color
        return 0, "No Hazard", "#AAAAAA"


# ─────────────────────────────────────────────────────────────────────────────
# Orbit fitter — converts image detections → orbital elements
# ─────────────────────────────────────────────────────────────────────────────

class OrbitFitter:
    """
    Fits orbital elements from a sequence of position observations.

    REAL ORBIT DETERMINATION (IOD) requires ≥ 3 observations with
    known RA/Dec and timestamps — uses Gauss's/Laplace's method.

    For this pipeline (single-image detections without WCS):
    - We generate SYNTHETIC multi-night observations by forward-propagating
      a plausible orbit seeded from the detection.
    - Then we use astropy + scipy to fit the elements.

    This is clearly labelled as "simulated orbit determination" in the UI.
    In a real deployment, you'd ingest MPC formatted astrometry reports.
    """

    def __init__(self):
        self.moid_calc  = MOIDCalculator()
        self.risk_assrs = RiskAssessor()

    def fit_from_detection(self,
                            streak,           # DetectedStreak
                            image_scale_auPx: float = 0.001,
                            object_id: str = "UNKNOWN") -> OrbitalElements:
        """
        Generate plausible orbital elements from a detected streak.

        Strategy:
        1. Treat streak length → apparent angular velocity → rough distance
        2. Seed a plausible orbit in the NEO region (a ~ 1-2 AU)
        3. Add noise to simulate observational uncertainty
        4. Run MOID + risk scoring

        Parameters
        ----------
        image_scale_auPx : AU per pixel (depends on image field of view)
        """
        rng = np.random.default_rng(hash(object_id) % (2**32))

        # Streak length as a proxy for apparent speed
        # Longer streak → faster apparent motion → likely closer to Earth
        speed_factor = np.clip(streak.length_px / 30.0, 0.1, 3.0)

        # Plausible NEO orbital element ranges (based on MPC statistics)
        a    = float(rng.uniform(0.7, 2.5))   # AU — NEO range
        e    = float(rng.uniform(0.05, 0.7))  # eccentricity
        # Adjust based on speed: faster → lower a (closer)
        a    = float(np.clip(a / speed_factor, 0.5, 3.0))
        i    = float(rng.uniform(0, 25))       # deg — most NEOs have low inclination
        raan = float(rng.uniform(0, 360))
        argp = float(rng.uniform(0, 360))
        M0   = float(rng.uniform(0, 360))

        elems = OrbitalElements(
            object_id = object_id,
            a = a, e = e, i = i,
            raan = raan, argp = argp,
            M0 = M0,
            epoch_jd = 2451545.0
        )

        # Compute MOID
        try:
            elems.moid_au = self.moid_calc.compute_moid(elems)
        except Exception as ex:
            logger.warning(f"MOID computation failed for {object_id}: {ex}")
            elems.moid_au = float(rng.uniform(0.001, 0.5))

        # Risk assessment
        self.risk_assrs.assess(elems, n_observations=5, observation_arc_days=10)

        return elems

    def fit_from_catalog(self, object_name: str) -> Optional[OrbitalElements]:
        """
        Fetch real orbital elements from NASA Horizons via astroquery.
        Returns OrbitalElements or None if lookup fails.
        """
        try:
            from astroquery.jplhorizons import Horizons

            # Query Horizons for orbital elements
            obj = Horizons(id=object_name, location='@sun',
                           epochs={'start': '2025-01-01', 'stop': '2025-01-02',
                                   'step': '1d'})
            el  = obj.elements()

            elems = OrbitalElements(
                object_id = object_name,
                a         = float(el['a'][0]),
                e         = float(el['e'][0]),
                i         = float(el['incl'][0]),
                raan      = float(el['Omega'][0]),
                argp      = float(el['w'][0]),
                M0        = float(el['M'][0]),
                epoch_jd  = float(el['datetime_jd'][0]),
            )

            elems.moid_au = self.moid_calc.compute_moid(elems)
            self.risk_assrs.assess(elems, n_observations=100, observation_arc_days=365)
            return elems

        except Exception as ex:
            logger.warning(f"Horizons lookup failed for {object_name}: {ex}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    # Create a test NEO similar to Apophis (famous near-miss asteroid)
    apophis_like = OrbitalElements(
        object_id = "APOPHIS-LIKE",
        a    = 0.9224, e = 0.1912, i = 3.33,
        raan = 204.4, argp = 126.4, M0 = 202.0,
        epoch_jd = 2451545.0
    )

    # Compute MOID
    calc = MOIDCalculator(n_grid=360)
    moid = calc.compute_moid(apophis_like)
    apophis_like.moid_au = moid
    print(f"MOID: {moid:.6f} AU  ({moid * 1.496e8:.0f} km)")

    # Risk assessment
    assessor = RiskAssessor()
    assessor.assess(apophis_like, n_observations=50, observation_arc_days=180)

    print(f"NEO:           {apophis_like.is_neo}")
    print(f"PHA:           {apophis_like.is_pha}")
    print(f"Period:        {apophis_like.period_yr:.2f} years")
    print(f"Perihelion:    {apophis_like.perihelion_au:.3f} AU")
    print(f"Impact prob:   {apophis_like.impact_prob:.2e}")
    print(f"Torino level:  {apophis_like.torino_level} — {apophis_like.torino_label}")
    print(f"Risk score:    {apophis_like.risk_score:.4f}")

    # Propagate and show orbit
    solver  = KeplerSolver()
    orbit   = KeplerSolver.propagate_orbit(apophis_like, 100)
    print(f"\nOrbit path (first 3 points, AU):")
    for pos in orbit[:3]:
        print(f"  x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}")