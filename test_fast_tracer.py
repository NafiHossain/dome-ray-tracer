"""Checks the tracer against answers that come from exact formulas, not from the tracer."""
import math

import numpy as np
import pytest

from fast_tracer import fresnel, trace

RAYS = 400_000


def test_fresnel_known_values():
    assert fresnel(np.array([1.0]), 1.5)[0] == pytest.approx(0.04)  # ((n - 1) / (n + 1))^2
    critical = math.asin(1 / 1.5)
    assert fresnel(np.array([math.cos(critical + 0.01)]), 1.5)[0] == 1.0  # total internal reflection
    assert fresnel(np.array([math.cos(critical - 1e-6)]), 1.5)[0] > 0.9  # it rises to 1 at the critical angle
    brewster = math.atan(1 / 1.5)  # r_p = 0 here, so the unpolarized value is r_s / 2
    cos_i, cos_t = math.cos(brewster), math.sqrt(1 - (1.5 * math.sin(brewster)) ** 2)
    r_s = ((1.5 * cos_i - cos_t) / (1.5 * cos_i + cos_t)) ** 2
    assert fresnel(np.array([cos_i]), 1.5)[0] == pytest.approx(r_s / 2)


@pytest.mark.parametrize("n", [1.5, 2.4])
@pytest.mark.parametrize("absorb", [0.1, 1.0])
def test_point_emitter_at_the_dome_centre_meets_each_surface_at_normal_incidence(n, absorb):
    # A ray goes out along a radius and reflects straight back to the emitter: T + R (1 - A) T + ... = T / (1 - R (1 - A)).
    reflect = ((n - 1) / (n + 1)) ** 2
    expected = (1 - reflect) / (1 - reflect * (1 - absorb))
    result = trace("dome", n=n, rays=RAYS, emitter_area=1e-10, emitter_absorption=absorb)
    assert result.extracted == pytest.approx(expected, abs=0.002)
    assert result.upward == result.extracted and result.trapped == 0


@pytest.mark.parametrize("n", [1.4, 1.5, 2.0, 3.2])
def test_a_perfect_box_traps_every_ray_outside_all_six_escape_cones(n):
    # Specular reflection in a box only flips the signs of the direction cosines, so a ray's class never changes.
    rng = np.random.default_rng(99)
    phi = rng.uniform(0, 2 * np.pi, 4_000_000)
    z = np.sqrt(rng.uniform(size=4_000_000))
    x, y = np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi)
    limit = math.sqrt(1 - 1 / n ** 2)
    expected = np.mean((np.abs(x) < limit) & (np.abs(y) < limit) & (z < limit))

    result = trace("slab", n=n, rays=50_000, emitter_absorption=0.0, slab_side=2.0, slab_thickness=1.0, emitter_area=0.5,
                   max_bounces=300)
    assert result.trapped / result.rays == pytest.approx(expected, abs=0.012)  # 5 sigma for 50 000 rays


def test_a_very_wide_slab_matches_the_two_surface_fresnel_series():
    # Rays inside the escape cone: top = T / (1 - R^2), bottom = R T / (1 - R^2), integrated over Lambertian emission.
    n = 1.5
    sin2 = np.linspace(0, 1 / n ** 2, 200_001)[:-1]
    reflect = fresnel(np.sqrt(1 - sin2), n)
    weight = np.gradient(sin2)
    top = np.sum(weight * (1 - reflect) / (1 - reflect ** 2))
    bottom = np.sum(weight * reflect * (1 - reflect) / (1 - reflect ** 2))
    result = trace("slab", n=n, rays=RAYS, slab_side=1e9, slab_thickness=1.0, emitter_area=1e-6, emitter_absorption=0.0,
                   max_bounces=200)
    assert result.top / RAYS == pytest.approx(top, abs=0.002)
    assert result.bottom / RAYS == pytest.approx(bottom, abs=0.002)


def test_bulk_absorption_follows_beer_lambert():
    # Index matching (n = outside) removes reflections, so every ray crosses exactly one radius of glass.
    alpha, radius = 0.2, 5.0
    result = trace("dome", n=1.5, outside_n=1.5, rays=RAYS, dome_radius=radius, emitter_area=1e-10,
                   bulk_absorption=alpha)
    assert result.extracted == pytest.approx(math.exp(-alpha * radius), abs=0.002)


def test_index_matching_lets_every_ray_out_at_the_first_surface():
    result = trace("slab", n=1.5, outside_n=1.5, rays=50_000, emitter_absorption=1.0)
    assert result.extracted == 1.0 and result.emitter_absorbed == 0


def test_a_mirror_base_sends_no_ray_out_of_the_bottom():
    result = trace("dome", n=2.4, rays=50_000, base="mirror", emitter_area=math.pi)
    assert result.bottom == 0 and result.top > 0


@pytest.mark.parametrize("shape", ["dome", "slab"])
def test_every_ray_ends_in_exactly_one_group(shape):
    r = trace(shape, n=2.4, rays=50_000, bulk_absorption=0.05, scatter=0.5)
    assert r.top + r.side + r.bottom + r.emitter_absorbed + r.bulk_absorbed + r.trapped == r.rays


def test_a_dome_traps_light_only_when_the_emitter_breaks_the_no_tir_limit():
    # Emitter radius / dome radius < 1 / n means that no ray can meet the dome above the critical angle.
    n, emitter_radius = 2.4, 1.0
    safe = trace("dome", n=n, rays=100_000, dome_radius=1.05 * n * emitter_radius, emitter_absorption=1.0)
    broken = trace("dome", n=n, rays=100_000, dome_radius=0.6 * n * emitter_radius, emitter_absorption=1.0)
    assert safe.extracted > 0.82 > broken.extracted


def test_the_same_seed_gives_the_same_result():
    assert trace("dome", n=2.4, rays=10_000, seed=4).top == trace("dome", n=2.4, rays=10_000, seed=4).top
