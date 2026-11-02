#!/usr/bin/env python3
"""Fast Monte Carlo ray tracer for light extraction from a glass dome or slab.

It moves all rays at once with NumPy array operations. Both shapes are convex, so the distance to the next surface has
an exact formula and a ray needs no small nudge off a surface.

Physics:
- Emitter: uniform on the base, with Lambertian (cosine) emission into the glass.
- Each glass surface: unpolarized Fresnel reflection, with total internal reflection above the critical angle.
- Glass: Beer-Lambert bulk absorption.
- Rough surfaces (dome surface, slab top): a reflected ray scatters in a Lambertian direction with probability `scatter`.
- Emitter on the way back: absorbs with probability `emitter_absorption`, otherwise it reflects like a mirror.
- Rest of the base: bare glass, or a mirror if `base="mirror"`.

Ray optics only: no wave effects, a single wavelength and no change of index with wavelength.
"""
import argparse
import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    rays: int
    top: int = 0  # left through the dome surface or the slab top
    side: int = 0  # left through a slab side
    bottom: int = 0  # left through the base
    emitter_absorbed: int = 0
    bulk_absorbed: int = 0
    trapped: int = 0  # still inside after max_bounces
    seconds: float = 0.0

    @property
    def extracted(self) -> float:
        """Escaped rays / all rays. A trapped ray counts as not extracted."""
        return (self.top + self.side + self.bottom) / self.rays

    @property
    def upward(self) -> float:
        return self.top / self.rays

    @property
    def error(self) -> float:
        return math.sqrt(max(self.extracted * (1 - self.extracted), 1e-12) / self.rays)


def fresnel(cos_i: np.ndarray, n: float, outside: float = 1.0) -> np.ndarray:
    """Unpolarized power reflectance from glass (n) into the outside medium. Returns 1 for total internal reflection."""
    sin_t2 = (n / outside) ** 2 * (1 - cos_i ** 2)
    out = np.ones_like(cos_i)
    ok = sin_t2 < 1
    ci, ct = cos_i[ok], np.sqrt(1 - sin_t2[ok])
    r_s = ((n * ci - outside * ct) / (n * ci + outside * ct)) ** 2
    r_p = ((n * ct - outside * ci) / (n * ct + outside * ci)) ** 2
    out[ok] = (r_s + r_p) / 2
    return out


def lambertian(normal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cosine-weighted directions around each unit normal."""
    phi = rng.uniform(0, 2 * np.pi, len(normal))
    cos_t = np.sqrt(rng.uniform(size=len(normal)))
    sin_t = np.sqrt(1 - cos_t ** 2)
    helper = np.where(np.abs(normal[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(normal, u)
    return (u * (sin_t * np.cos(phi))[:, None] + v * (sin_t * np.sin(phi))[:, None] + normal * cos_t[:, None])


def next_dome(pos: np.ndarray, d: np.ndarray, radius: float):
    """Exact distance, surface normal and surface name (0 curved, 2 base) from inside a hemisphere with its base at z = 0."""
    b = np.sum(pos * d, axis=1)
    c = np.sum(pos * pos, axis=1) - radius ** 2
    t = -b + np.sqrt(np.maximum(b * b - c, 0))  # the far root: the inside point has one forward hit on the sphere
    down = d[:, 2] < 0
    t_base = np.full(len(pos), np.inf)
    t_base[down] = -pos[down, 2] / d[down, 2]
    base = t_base < t
    t = np.where(base, t_base, t)
    hit = pos + d * t[:, None]
    normal = hit / np.linalg.norm(hit, axis=1, keepdims=True)
    normal[base] = [0.0, 0.0, -1.0]
    return t, normal, np.where(base, 2, 0)


def next_slab(pos: np.ndarray, d: np.ndarray, side: float, thickness: float):
    """Exact distance, surface normal and surface name (0 top, 1 side, 2 base) from inside a box with its base at z = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t_x = np.where(d[:, 0] > 0, (side / 2 - pos[:, 0]) / d[:, 0], (-side / 2 - pos[:, 0]) / d[:, 0])
        t_y = np.where(d[:, 1] > 0, (side / 2 - pos[:, 1]) / d[:, 1], (-side / 2 - pos[:, 1]) / d[:, 1])
        t_z = np.where(d[:, 2] > 0, (thickness - pos[:, 2]) / d[:, 2], -pos[:, 2] / d[:, 2])
    all_t = np.nan_to_num(np.stack([t_x, t_y, t_z], axis=1), nan=np.inf, posinf=np.inf)
    axis = np.argmin(all_t, axis=1)
    normal = np.zeros_like(pos)
    normal[np.arange(len(pos)), axis] = np.sign(d[np.arange(len(pos)), axis])
    name = np.where(axis == 2, np.where(d[:, 2] > 0, 0, 2), 1)
    return all_t[np.arange(len(pos)), axis], normal, name


def trace(shape: str, n: float = 1.5, rays: int = 200_000, dome_radius: float = 5.0, slab_side: float = 10.0,
          slab_thickness: float = 3.0, emitter_area: float = math.pi, emitter_absorption: float = 0.1,
          bulk_absorption: float = 0.0, scatter: float = 0.0, outside_n: float = 1.0, base: str = "glass",
          max_bounces: int = 500, seed: int = 0) -> Result:
    """Trace `rays` rays through a "dome" or a "slab". Lengths are in mm and bulk_absorption is in 1/mm."""
    if shape not in ("dome", "slab") or base not in ("glass", "mirror"):
        raise ValueError("shape must be dome or slab, and base must be glass or mirror")
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    result = Result(rays=rays)
    emitter_radius = math.sqrt(emitter_area / math.pi)
    half_side = math.sqrt(emitter_area) / 2

    def on_emitter(p: np.ndarray) -> np.ndarray:
        if shape == "dome":
            return p[:, 0] ** 2 + p[:, 1] ** 2 <= emitter_radius ** 2
        return (np.abs(p[:, 0]) <= half_side) & (np.abs(p[:, 1]) <= half_side)

    if shape == "dome":
        if emitter_radius >= dome_radius:
            raise ValueError("the emitter must fit on the dome base")
        r = emitter_radius * np.sqrt(rng.uniform(size=rays))
        angle = rng.uniform(0, 2 * np.pi, rays)
        pos = np.column_stack([r * np.cos(angle), r * np.sin(angle), np.zeros(rays)])
    else:
        if 2 * half_side >= slab_side:
            raise ValueError("the emitter must fit on the slab base")
        pos = np.column_stack([rng.uniform(-half_side, half_side, (rays, 2)), np.zeros(rays)])
    up = np.tile([0.0, 0.0, 1.0], (rays, 1))
    d = lambertian(up, rng)

    for _ in range(max_bounces):
        if len(pos) == 0:
            break
        if shape == "dome":
            t, normal, surface = next_dome(pos, d, dome_radius)
        else:
            t, normal, surface = next_slab(pos, d, slab_side, slab_thickness)

        if bulk_absorption > 0:
            absorbed = rng.exponential(1 / bulk_absorption, len(pos)) < t
            result.bulk_absorbed += int(absorbed.sum())
            keep = ~absorbed
            pos, d, t, normal, surface = pos[keep], d[keep], t[keep], normal[keep], surface[keep]
        pos = pos + d * t[:, None]
        cos_i = np.abs(np.sum(d * normal, axis=1))

        base_hit = surface == 2
        emitter_hit = base_hit & on_emitter(pos)
        emitter_takes = emitter_hit & (rng.uniform(size=len(pos)) < emitter_absorption)
        result.emitter_absorbed += int(emitter_takes.sum())

        reflect_p = fresnel(cos_i, n, outside_n)
        reflect_p[emitter_hit] = 1.0
        if base == "mirror":
            reflect_p[base_hit] = 1.0
        leaves = (rng.uniform(size=len(pos)) >= reflect_p) & ~emitter_takes
        result.top += int((leaves & (surface == 0)).sum())
        result.side += int((leaves & (surface == 1)).sum())
        result.bottom += int((leaves & base_hit).sum())

        keep = ~(leaves | emitter_takes)
        pos, d, normal, surface = pos[keep], d[keep], normal[keep], surface[keep]
        d = d - 2 * np.sum(d * normal, axis=1, keepdims=True) * normal
        d /= np.linalg.norm(d, axis=1, keepdims=True)  # stops rounding error from growing over many bounces
        if scatter > 0:
            rough = (surface == 0) & (rng.uniform(size=len(pos)) < scatter)
            if rough.any():
                d[rough] = lambertian(-normal[rough], rng)

    result.trapped = len(pos)
    result.seconds = time.perf_counter() - start
    return result


def sweep(label: str, values: list, run) -> list:
    print(f"\n{label}\n{'value':>7} | {'dome extracted':>15} {'upward':>7} {'trapped':>8} | "
          f"{'slab extracted':>15} {'upward':>7} {'trapped':>8}")
    rows = []
    for value in values:
        dome, slab = run("dome", value), run("slab", value)
        rows.append((value, dome, slab))
        print(f"{value:7.2f} | {dome.extracted:7.4f} +-{dome.error:.4f} {dome.upward:7.3f} {dome.trapped / dome.rays:8.3f} | "
              f"{slab.extracted:7.4f} +-{slab.error:.4f} {slab.upward:7.3f} {slab.trapped / slab.rays:8.3f}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rays", type=int, default=50_000)
    parser.add_argument("--plot", default="dome_vs_slab.png", help="output image; empty for no plot")
    args = parser.parse_args()

    start = time.perf_counter()
    common = {"rays": args.rays, "emitter_absorption": 0.1}
    by_index = sweep("Refractive index (dome radius 5 mm; slab 10 x 10 x 3 mm; emitter area pi mm2)",
                     [1.4, 1.5, 1.6, 1.8, 2.0, 2.4, 3.2], lambda shape, v: trace(shape, n=v, **common))
    by_size = sweep("Size at n = 2.4 (dome radius = S; slab side = 2 S, thickness = 0.6 S; S in mm)",
                    [1.5, 2.0, 3.0, 5.0, 8.0], lambda shape, v: trace(
                        shape, n=2.4, dome_radius=v, slab_side=2 * v, slab_thickness=0.6 * v, **common))
    by_loss = sweep("Bulk absorption in 1/mm at n = 2.4", [0.0, 0.01, 0.03, 0.1, 0.3],
                    lambda shape, v: trace(shape, n=2.4, bulk_absorption=v, **common))
    by_scatter = sweep("Surface scatter probability at n = 2.4", [0.0, 0.25, 0.5, 0.75, 1.0],
                       lambda shape, v: trace(shape, n=2.4, scatter=v, **common))
    print(f"\n{args.rays} rays per point. Total time: {time.perf_counter() - start:.1f} s")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panels = ((by_index, "Refractive index"), (by_size, "Size S (mm)"),
                  (by_loss, "Bulk absorption (1/mm)"), (by_scatter, "Surface scatter probability"))
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
        for ax, (rows, label) in zip(axes, panels):
            x = [row[0] for row in rows]
            ax.plot(x, [row[1].extracted for row in rows], "o-", label="Dome")
            ax.plot(x, [row[2].extracted for row in rows], "s-", label="Slab")
            ax.set_xlabel(label)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Escaped rays / all rays")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)


if __name__ == "__main__":
    main()
