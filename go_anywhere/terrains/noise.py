"""Band-limited value-noise roughness for heightfields.

Cheap enough to run per cell at generation time (pure numpy, O(pixels)).
"""

from __future__ import annotations

import numpy as np


def _value_noise(
  shape: tuple[int, int], period_px: float, rng: np.random.Generator
) -> np.ndarray:
  """Bilinearly interpolated random lattice noise in [-1, 1]."""
  ny, nx = shape
  gy = int(np.ceil(ny / period_px)) + 2
  gx = int(np.ceil(nx / period_px)) + 2
  lattice = rng.uniform(-1.0, 1.0, size=(gy, gx))
  ys = np.arange(ny) / period_px
  xs = np.arange(nx) / period_px
  y0 = np.floor(ys).astype(int)
  x0 = np.floor(xs).astype(int)
  ty = (ys - y0)[:, None]
  tx = (xs - x0)[None, :]
  # Smoothstep for C1 continuity.
  ty = ty * ty * (3 - 2 * ty)
  tx = tx * tx * (3 - 2 * tx)
  a = lattice[y0][:, x0]
  b = lattice[y0][:, x0 + 1]
  c = lattice[y0 + 1][:, x0]
  d = lattice[y0 + 1][:, x0 + 1]
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


def fbm(
  shape: tuple[int, int],
  rng: np.random.Generator,
  base_period_px: float,
  octaves: int = 3,
  persistence: float = 0.5,
  lacunarity: float = 2.0,
) -> np.ndarray:
  """Fractal Brownian motion from value noise, normalized to [-1, 1]."""
  out = np.zeros(shape, dtype=np.float64)
  amp, period, total = 1.0, base_period_px, 0.0
  for _ in range(octaves):
    out += amp * _value_noise(shape, max(1.0, period), rng)
    total += amp
    amp *= persistence
    period /= lacunarity
  return out / max(total, 1e-9)


def band_limited_roughness(
  shape: tuple[int, int],
  resolution: float,
  rng: np.random.Generator,
  amplitude: float,
  min_wavelength: float = 0.2,
  max_wavelength: float = 0.4,
) -> np.ndarray:
  """Roughness concentrated in a wavelength band (meters), peak |h| ≈ amplitude."""
  if amplitude <= 0.0:
    return np.zeros(shape, dtype=np.float64)
  lo = max(1.0, min_wavelength / resolution)
  hi = max(lo, max_wavelength / resolution)
  n = fbm(shape, rng, base_period_px=hi, octaves=2, lacunarity=hi / lo)
  return amplitude * n
