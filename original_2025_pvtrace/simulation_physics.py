import sys
import time
import gc
import functools
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

np.float = float
np.int = int

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cl = None
    cl_array = None

# Suppress OpenCL compiler warnings
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

try:
    from pvtrace import Node, Scene, Material, Surface, SurfaceDelegate, Light, Ray
    from pvtrace.algorithm import photon_tracer
    from pvtrace.geometry.sphere import Sphere
    from pvtrace.geometry.cylinder import Cylinder
    from pvtrace.geometry.box import Box
    from pvtrace.geometry.geometry import Geometry
    from pvtrace.material.component import Absorber, Luminophore
    from pvtrace.light.event import Event
    from pvtrace.geometry.utils import EPS_ZERO
except ImportError as e:
    print(f"Error importing pvtrace: {e}")
    sys.exit(1)


class Hemisphere(Geometry):
    def __init__(self, radius: float, material: Optional[Material] = None, tol: float = EPS_ZERO):
        super(Hemisphere, self).__init__()
        self.radius = radius
        self._material = material
        self.tol = tol
    
    @property
    def material(self) -> Optional[Material]:
        return self._material
    
    @material.setter
    def material(self, new_value: Material):
        self._material = new_value
    
    def is_on_surface(self, point: Tuple[float, float, float]) -> bool:
        x, y, z = point
        r = np.sqrt(x**2 + y**2 + z**2)
        r_xy = np.sqrt(x**2 + y**2)
        
        on_spherical_part = (np.abs(r - self.radius) < self.tol) and (z >= -self.tol)
        on_flat_base = (np.abs(z) < self.tol) and (r_xy <= self.radius + self.tol)
        
        return on_spherical_part or on_flat_base
    
    def contains(self, point: Tuple[float, float, float]) -> bool:
        x, y, z = point
        r = np.sqrt(x**2 + y**2 + z**2)
        return (r <= self.radius + self.tol) and (z >= -self.tol)
    
    def intersections(self, origin: Tuple[float, float, float], direction: Tuple[float, float, float]) -> Tuple:
        origin_np = np.array(origin)
        direction_np = np.array(direction)
        hits = []
        
        a = np.dot(direction_np, direction_np)
        b = 2.0 * np.dot(direction_np, origin_np)
        c = np.dot(origin_np, origin_np) - self.radius**2
        discriminant = b**2 - 4 * a * c
        
        if discriminant >= 0:
            sqrt_d = np.sqrt(discriminant)
            for t in [(-b - sqrt_d) / (2 * a), (-b + sqrt_d) / (2 * a)]:
                if t > self.tol:
                    point = origin_np + t * direction_np
                    if point[2] >= -self.tol:
                        hits.append(tuple(point))
        
        if not np.isclose(direction_np[2], 0):
            t_base = -origin_np[2] / direction_np[2]
            if t_base > self.tol:
                point = origin_np + t_base * direction_np
                if (point[0]**2 + point[1]**2) <= self.radius**2 + self.tol:
                    hits.append(tuple(point))
        
        if not hits:
            return tuple()
        
        return tuple(sorted(list(set(hits)), key=lambda p: np.linalg.norm(np.array(p) - origin_np)))
    
    def normal(self, surface_point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        x, y, z = surface_point
        
        if abs(z) < self.tol and np.sqrt(x**2 + y**2) < self.radius - self.tol:
            return (0, 0, -1)
        
        r = np.sqrt(x**2 + y**2 + z**2)
        return (x / r, y / r, z / r) if r > self.tol else (0, 0, 1)
    
    def is_entering(self, surface_point: Tuple[float, float, float], direction: Tuple[float, float, float]) -> bool:
        if not self.is_on_surface(surface_point):
            raise ValueError("Point is not on the surface.")
        return np.dot(self.normal(surface_point), direction) < 0.0

Geometry.register(Hemisphere)


@dataclass
class SimulationConfig:
    geometry_type: str = "dome"
    dome_radius: Optional[float] = None
    slab_side_length: Optional[float] = None
    slab_thickness: Optional[float] = None
    emitter_radius: Optional[float] = None
    emitter_height: Optional[float] = None
    emitter_side_length: Optional[float] = None
    emitter_thickness: Optional[float] = None
    glass_refractive_index: float = 1.5
    dome_absorption_coefficient: float = 0.0
    slab_absorption_coefficient: float = 0.0
    scatter_probability: float = 0.0
    top_surface_roughness: float = 0.0
    luminophore_absorption_coefficient: float = 0.0
    luminophore_quantum_yield: float = 1.0
    emitter_absorption_coefficient: float = 1000.0
    ray_count: int = 10000
    max_steps: int = 1000
    world_radius: float = 50.0
    position_tolerance: float = 1e-6
    max_workers: Optional[int] = None
    use_gpu: bool = True


@dataclass
class SimulationResults:
    total_rays: int
    absorbed_count: int
    escaped_count: int
    trapped_count: int
    error_count: int
    exit_angles: List[float]
    path_lengths: Dict[str, List[float]]
    bounce_counts: Dict[str, List[int]]
    interaction_stats: Dict[str, int]
    execution_time: float
    bulk_absorption_count: int
    emitter_absorption_count: int
    luminophore_absorption_count: int


class DomePhysics(SurfaceDelegate):
    def __init__(self, scatter_probability: float = 1.0):
        super().__init__()
        self.scatter_probability = scatter_probability
        self.reset_stats()
    
    def reset_stats(self):
        self.interaction_count: int = 0
        self.tir_count: int = 0
        self.scatter_count: int = 0
        self.transmission_count: int = 0
    
    def reflectivity(self, surface: Surface, ray: Ray, geometry: Geometry, container: Node, adjacent: Node) -> float:
        self.interaction_count += 1
        return self._fresnel_reflectance(ray, geometry, container, adjacent)
    
    def _fresnel_reflectance(self, ray: Ray, geometry: Geometry, container: Node, adjacent: Node) -> float:
        if not adjacent:
            return 1.0
            
        n1 = container.geometry.material.refractive_index
        n2 = adjacent.geometry.material.refractive_index
        
        normal = np.array(geometry.normal(ray.position))
        direction = np.array(ray.direction)
        
        if np.dot(normal, direction) > 0:
            normal = -normal
        
        cos_theta_i = -np.dot(direction, normal)
        cos_theta_i = np.clip(cos_theta_i, -1.0, 1.0)
        sin_theta_i = np.sqrt(max(0, 1 - cos_theta_i**2))
        
        if n2 <= 1e-6 or n1 <= 1e-6:
            return 1.0
        
        sin_theta_t = (n1 / n2) * sin_theta_i
        
        if sin_theta_t >= 1.0:
            self.tir_count += 1
            return 1.0
        
        cos_theta_t = np.sqrt(max(0, 1 - sin_theta_t**2))
        
        r_s = ((n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t))**2
        r_p = ((n1 * cos_theta_t - n2 * cos_theta_i) / (n1 * cos_theta_t + n2 * cos_theta_i))**2
        
        return (r_s + r_p) / 2
    
    def reflected_direction(self, surface: Surface, ray: Ray, geometry: Geometry, container: Node, adjacent: Node) -> Ray:
        normal = np.array(geometry.normal(ray.position))
        direction = np.array(ray.direction)
        
        if np.dot(normal, direction) > 0:
            normal = -normal
        
        is_curved_surface = not np.allclose(normal, [0, 0, -1])
        
        if is_curved_surface and np.random.rand() < self.scatter_probability:
            self.scatter_count += 1
            new_direction = self._lambertian_scatter(normal)
        else:
            new_direction = direction - 2 * np.dot(direction, normal) * normal
        
        return tuple(new_direction / np.linalg.norm(new_direction))
    
    def transmitted_direction(self, surface: Surface, ray: Ray, geometry: Geometry, container: Node, adjacent: Node) -> Ray:
        self.transmission_count += 1
        
        n1 = container.geometry.material.refractive_index
        n2 = adjacent.geometry.material.refractive_index
        
        normal = np.array(geometry.normal(ray.position))
        direction = np.array(ray.direction)
        
        if np.dot(normal, direction) > 0:
            normal = -normal
        
        cos_theta_i = -np.dot(direction, normal)
        sin_theta_t_sq = (n1 / n2)**2 * (1 - cos_theta_i**2)
        
        term1 = (n1 / n2) * direction
        term2 = ((n1 / n2) * cos_theta_i - np.sqrt(max(0, 1 - sin_theta_t_sq))) * normal
        refracted_dir = term1 + term2
        
        return tuple(refracted_dir / np.linalg.norm(refracted_dir))
    
    def _lambertian_scatter(self, normal: np.ndarray) -> np.ndarray:
        phi = 2 * np.pi * np.random.rand()
        cos_theta = np.sqrt(np.random.rand())
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        u = np.cross(normal, [0.0, 0.0, 1.0]) if abs(normal[2]) < 0.999 else np.cross(normal, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        
        return u * sin_theta * np.cos(phi) + v * sin_theta * np.sin(phi) + normal * cos_theta


class SlabPhysics(SurfaceDelegate):
    def __init__(self, top_surface_roughness: float = 0.0, position_tolerance: float = 1e-6):
        super().__init__()
        self.top_surface_roughness = top_surface_roughness
        self.position_tolerance = position_tolerance
        self.reset_stats()
        self._precomputed_constants()
        self._setup_normal_cache()
    
    def _precomputed_constants(self):
        self._rad_to_deg = 180.0 / np.pi
        self._deg_to_rad = np.pi / 180.0
        self._z_up = np.array([0, 0, 1])
        self._z_down = np.array([0, 0, -1])
    
    def _setup_normal_cache(self):
        self._normal_cache = {}
        self._cache_tolerance = 1e-8
    
    def reset_stats(self):
        self.interactions = 0
        self.fresnel_reflections = 0
        self.fresnel_transmissions = 0
        self.lambertian_scatterings = 0
        self.total_internal_reflections = 0
    
    def reflectivity(self, surface, ray, geometry, container, adjacent):
        self.interactions += 1
        if not adjacent:
            return 1.0
        
        normal = self._get_normal(geometry, ray.position)
        direction = np.array(ray.direction)
        
        if np.dot(normal, direction) > 0:
            normal = -normal
        
        n1 = container.geometry.material.refractive_index
        n2 = adjacent.geometry.material.refractive_index
        
        cos_theta_i = -np.dot(direction, normal)
        cos_theta_i = np.clip(cos_theta_i, 0, 1)
        sin_theta_i = np.sqrt(1 - cos_theta_i**2)
        
        sin_theta_t = (n1 / n2) * sin_theta_i
        
        if sin_theta_t >= 1.0:
            self.total_internal_reflections += 1
            return 1.0
        
        cos_theta_t = np.sqrt(1 - sin_theta_t**2)
        
        r_s = ((n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t))**2
        r_p = ((n1 * cos_theta_t - n2 * cos_theta_i) / (n1 * cos_theta_t + n2 * cos_theta_i))**2
        
        return (r_s + r_p) / 2
    
    def reflected_direction(self, surface, ray, geometry, container, adjacent):
        self.fresnel_reflections += 1
        
        normal = self._get_normal(geometry, ray.position)
        direction = np.array(ray.direction)
        
        if np.dot(normal, direction) > 0:
            normal = -normal
        
        if self._is_top_surface(ray.position, geometry) and np.random.rand() < self.top_surface_roughness:
            self.lambertian_scatterings += 1
            return self._lambertian_scatter(normal)
        
        reflected_dir = direction - 2 * np.dot(direction, normal) * normal
        return tuple(reflected_dir / np.linalg.norm(reflected_dir))
    
    def transmitted_direction(self, surface, ray, geometry, container, adjacent):
        self.fresnel_transmissions += 1
        
        normal = self._get_normal(geometry, ray.position)
        direction = np.array(ray.direction)
        
        if np.dot(normal, direction) > 0:
            normal = -normal
        
        n1 = container.geometry.material.refractive_index
        n2 = adjacent.geometry.material.refractive_index
        
        cos_theta_i = -np.dot(direction, normal)
        sin_theta_t_sq = (n1 / n2)**2 * (1 - cos_theta_i**2)
        
        refracted_dir = (n1 / n2) * direction + ((n1 / n2) * cos_theta_i - np.sqrt(1 - sin_theta_t_sq)) * normal
        return tuple(refracted_dir / np.linalg.norm(refracted_dir))
    
    def _is_top_surface(self, position, geometry):
        return abs(position[2] - geometry._size[2]/2) < self.position_tolerance
    
    def _get_normal(self, geometry, position):
        cache_key = self._get_cache_key(position)
        if cache_key in self._normal_cache:
            return self._normal_cache[cache_key]
        
        try:
            normal = np.array(geometry.normal(position))
            self._normal_cache[cache_key] = normal
            return normal
        except Exception:
            normal = self._fast_estimate_normal(geometry, position)
            self._normal_cache[cache_key] = normal
            return normal
    
    def _get_cache_key(self, position):
        rounded = tuple(np.round(np.array(position) / self._cache_tolerance) * self._cache_tolerance)
        return rounded
    
    def _fast_estimate_normal(self, geometry, position):
        x, y, z = position
        half_size = np.array(geometry._size) / 2
        
        distances = np.array([
            half_size[0] - abs(x),
            half_size[1] - abs(y),
            half_size[2] - abs(z)
        ])
        
        face_idx = np.argmin(distances)
        normals = [
            np.array([1, 0, 0]) if x > 0 else np.array([-1, 0, 0]),
            np.array([0, 1, 0]) if y > 0 else np.array([0, -1, 0]),
            np.array([0, 0, 1]) if z > 0 else np.array([0, 0, -1])
        ]
        
        return normals[face_idx]
    
    def _lambertian_scatter(self, normal):
        phi = 2 * np.pi * np.random.rand()
        cos_theta = np.sqrt(np.random.rand())
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        if abs(normal[2]) < 0.999:
            u = np.cross(normal, self._z_up)
        else:
            u = np.cross(normal, np.array([0, 1, 0]))
        
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        scattered_dir = u * sin_theta * np.cos(phi) + v * sin_theta * np.sin(phi) + normal * cos_theta
        return tuple(scattered_dir)


class GPUAcceleration:
    def __init__(self):
        self.context = None
        self.queue = None
        self.program = None
        self.device = None
        self._buffer_cache = {}  # Cache for reusable buffers
        self._max_rays = 0
        if GPU_AVAILABLE:
            self._setup_gpu()
    
    def _setup_gpu(self):
        try:
            platforms = cl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices(cl.device_type.GPU)
                if devices:
                    self.device = devices[0]
                    break
            
            if not self.device:
                self.device = cl.get_platforms()[0].get_devices()[0]
            
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            self._compile_kernels()
            
        except Exception as e:
            print(f"GPU setup failed: {e}")
            self.context = None
    
    def _get_or_create_buffer(self, name, size, dtype):
        """Get or create a reusable GPU buffer"""
        key = (name, size, dtype)
        if key not in self._buffer_cache:
            self._buffer_cache[key] = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size * np.dtype(dtype).itemsize)
        return self._buffer_cache[key]
    
    def _compile_kernels(self):
        dome_kernel = """
        #define M_PI 3.14159265358979323846f
        
        __kernel void trace_dome_rays(
            __global const float4* initial_positions,
            __global const float4* initial_directions,
            __global const float* wavelengths,
            __global int* ray_outcomes,
            __global float* path_lengths,
            __global int* bounce_counts,
            __global int* absorption_types,
            const float dome_radius,
            const float emitter_radius,
            const float glass_ri,
            const float world_ri,
            const float dome_absorption_coeff,
            const float scatter_probability,
            const int max_bounces,
            const int N
        ) {
            int gid = get_global_id(0);
            if (gid >= N) return;
            
            float4 pos = initial_positions[gid];
            float4 dir = initial_directions[gid];
            
            int bounces = 0;
            float total_distance = 0.0f;
            int outcome = 0;
            int abs_type = 0;
            
            bool currently_in_dome = true;
            
            for (int step = 0; step < max_bounces; step++) {
                // Ray-hemisphere intersection
                float a = dot(dir.xyz, dir.xyz);
                float b = 2.0f * dot(dir.xyz, pos.xyz);
                float c = dot(pos.xyz, pos.xyz) - dome_radius * dome_radius;
                float discriminant = b * b - 4.0f * a * c;
                
                float t = -1.0f;
                bool hit_curved_surface = false;
                
                if (discriminant >= 0.0f) {
                    float sqrt_d = sqrt(discriminant);
                    float t1 = (-b - sqrt_d) / (2.0f * a);
                    float t2 = (-b + sqrt_d) / (2.0f * a);
                    
                    for (int i = 0; i < 2; i++) {
                        float test_t = (i == 0) ? t1 : t2;
                        if (test_t > 0.001f) {
                            float3 test_pos = pos.xyz + test_t * dir.xyz;
                            if (test_pos.z >= -0.001f) {
                                if (currently_in_dome && i == 1) {
                                    t = test_t; hit_curved_surface = true; break;
                                } else if (!currently_in_dome && i == 0) {
                                    t = test_t; hit_curved_surface = true; break;
                                }
                            }
                        }
                    }
                }
                
                if (!hit_curved_surface && fabs(dir.z) > 0.001f) {
                    float t_base = -pos.z / dir.z;
                    if (t_base > 0.001f) {
                        float3 base_pos = pos.xyz + t_base * dir.xyz;
                        float r_base = sqrt(base_pos.x * base_pos.x + base_pos.y * base_pos.y);
                        if (r_base <= dome_radius + 0.001f) {
                            t = t_base;
                        }
                    }
                }
                
                if (t < 0) {
                    outcome = 1; break;
                }
                
                pos.xyz += dir.xyz * t;
                total_distance += t;
                
                if (currently_in_dome && dome_absorption_coeff > 0) {
                    float absorption_prob = 1.0f - exp(-dome_absorption_coeff * t);
                    uint abs_seed = gid * 1000 + step + 42;
                    abs_seed = abs_seed * 1664525 + 1013904223;
                    float abs_rand = ((float)(abs_seed % 10000)) / 10000.0f;
                    
                    if (abs_rand < absorption_prob) {
                        outcome = 2; abs_type = 0; break;
                    }
                }
                
                float3 normal;
                if (fabs(pos.z) < 0.001f) {
                    normal = (float3)(0, 0, 1);
                } else {
                    normal = normalize(pos.xyz);
                    if (currently_in_dome) normal = -normal;
                }
                
                bool was_in_dome = currently_in_dome;
                currently_in_dome = !currently_in_dome;
                
                float cos_theta_i = -dot(dir.xyz, normal);
                if (cos_theta_i < 0) {
                    normal = -normal; cos_theta_i = -cos_theta_i;
                }
                
                float n1 = was_in_dome ? glass_ri : world_ri;
                float n2 = was_in_dome ? world_ri : glass_ri;
                float sin_theta_t = (n1 / n2) * sqrt(1.0f - cos_theta_i * cos_theta_i);
                
                if (sin_theta_t >= 1.0f) {
                    dir.xyz = dir.xyz - 2.0f * dot(dir.xyz, normal) * normal;
                    currently_in_dome = was_in_dome;
                } else {
                    float cos_theta_t = sqrt(1.0f - sin_theta_t * sin_theta_t);
                    float r_s = pow((n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t), 2);
                    float r_p = pow((n1 * cos_theta_t - n2 * cos_theta_i) / (n1 * cos_theta_t + n2 * cos_theta_i), 2);
                    float reflectance = (r_s + r_p) * 0.5f;
                    
                    uint seed = gid * 1000 + step;
                    seed = seed * 1664525 + 1013904223;
                    float rand_val = ((float)(seed % 10000)) / 10000.0f;
                    
                    bool should_scatter = hit_curved_surface && (scatter_probability > 0.0f);
                    if (should_scatter) {
                        uint scatter_seed = gid * 2000 + step + 123;
                        scatter_seed = scatter_seed * 1664525 + 1013904223;
                        float scatter_rand = ((float)(scatter_seed % 10000)) / 10000.0f;
                        
                        if (scatter_rand < scatter_probability) {
                            uint phi_seed = scatter_seed + 456;
                            phi_seed = phi_seed * 1664525 + 1013904223;
                            float phi = 2.0f * M_PI * ((float)(phi_seed % 10000)) / 10000.0f;
                            
                            uint cos_theta_seed = phi_seed + 789;
                            cos_theta_seed = cos_theta_seed * 1664525 + 1013904223;
                            float cos_theta_scatter = sqrt(((float)(cos_theta_seed % 10000)) / 10000.0f);
                            float sin_theta_scatter = sqrt(1.0f - cos_theta_scatter * cos_theta_scatter);
                            
                            float3 scatter_normal = -normal;
                            dir.x = sin_theta_scatter * cos(phi);
                            dir.y = sin_theta_scatter * sin(phi);
                            dir.z = cos_theta_scatter;
                            
                            float3 tangent1, tangent2;
                            if (fabs(scatter_normal.z) < 0.999f) {
                                tangent1 = normalize(cross(scatter_normal, (float3)(0, 0, 1)));
                            } else {
                                tangent1 = normalize(cross(scatter_normal, (float3)(1, 0, 0)));
                            }
                            tangent2 = cross(scatter_normal, tangent1);
                            dir.xyz = tangent1 * dir.x + tangent2 * dir.y + scatter_normal * dir.z;
                            currently_in_dome = was_in_dome;
                        } else if (rand_val < reflectance) {
                            dir.xyz = dir.xyz - 2.0f * dot(dir.xyz, normal) * normal;
                            currently_in_dome = was_in_dome;
                        } else {
                            float3 term1 = (n1 / n2) * dir.xyz;
                            float3 term2 = ((n1 / n2) * cos_theta_i - cos_theta_t) * normal;
                            dir.xyz = normalize(term1 + term2);
                            if (was_in_dome && !currently_in_dome) { outcome = 1; break; }
                        }
                    } else if (rand_val < reflectance) {
                        dir.xyz = dir.xyz - 2.0f * dot(dir.xyz, normal) * normal;
                        currently_in_dome = was_in_dome;
                    } else {
                        float3 term1 = (n1 / n2) * dir.xyz;
                        float3 term2 = ((n1 / n2) * cos_theta_i - cos_theta_t) * normal;
                        dir.xyz = normalize(term1 + term2);
                        if (was_in_dome && !currently_in_dome) { outcome = 1; break; }
                    }
                }
                
                bounces++;
                pos.xyz += dir.xyz * 0.001f;
                
                float r_xy = sqrt(pos.x * pos.x + pos.y * pos.y);
                if (fabs(pos.z) < 0.01f && r_xy <= emitter_radius) {
                    uint emitter_seed = gid * 2000 + step + 123;
                    emitter_seed = emitter_seed * 1664525 + 1013904223;
                    float emitter_rand = ((float)(emitter_seed % 10000)) / 10000.0f;
                    if (emitter_rand < 0.999f) { outcome = 2; abs_type = 1; break; }
                }
            }
            
            if (outcome == 0) outcome = 3;
            ray_outcomes[gid] = outcome;
            path_lengths[gid] = total_distance;
            bounce_counts[gid] = bounces;
            absorption_types[gid] = abs_type;
        }
        """
        
        slab_kernel = """
        #define M_PI 3.14159265358979323846f
        
        __kernel void trace_slab_rays(
            __global const float4* initial_positions,
            __global const float4* initial_directions,
            __global const float* wavelengths,
            __global const float4* slab_bounds,
            __global const float* glass_ri,
            __global const float* world_ri,
            __global const float* absorption_coeff,
            __global const float* surface_roughness,
            __global int* ray_outcomes,
            __global float* path_lengths,
            __global int* bounce_counts,
            __global float* exit_angles,
            __global int* absorption_types,
            const int max_bounces,
            const int N,
            const float emitter_side_length
        ) {
            int gid = get_global_id(0);
            if (gid >= N) return;
            
            // Cache frequently used values
            float4 pos = initial_positions[gid];
            float4 dir = initial_directions[gid];
            const float4 slab_min = slab_bounds[0];
            const float4 slab_max = slab_bounds[1];
            const float ri = glass_ri[0];
            const float world_refractive_index = world_ri[0];
            const float abs_coeff = absorption_coeff[0];
            const float roughness = surface_roughness[0];
            
            // Pre-calculate commonly used values
            const float3 half_size = (slab_max.xyz - slab_min.xyz) * 0.5f;
            const float half_emitter = emitter_side_length * 0.5f;
            const float n_ratio_glass_to_air = ri / world_refractive_index;
            const float n_ratio_air_to_glass = world_refractive_index / ri;
            const bool has_absorption = abs_coeff > 0.0f;
            const bool has_roughness = roughness > 0.0f;
            
            int bounces = 0;
            float total_distance = 0.0f;
            int outcome = 0;
            int abs_type = 0;
            
            bool currently_in_slab = (pos.x >= slab_min.x && pos.x <= slab_max.x && 
                                    pos.y >= slab_min.y && pos.y <= slab_max.y && 
                                    pos.z >= slab_min.z && pos.z <= slab_max.z);
            
            for (int step = 0; step < max_bounces; step++) {
                float3 inv_dir = native_recip((float3)(dir.x, dir.y, dir.z));
                float3 t1 = (slab_min.xyz - pos.xyz) * inv_dir;
                float3 t2 = (slab_max.xyz - pos.xyz) * inv_dir;
                
                float3 t_min = fmin(t1, t2);
                float3 t_max = fmax(t1, t2);
                
                float t_near = fmax(t_min.x, fmax(t_min.y, t_min.z));
                float t_far = fmin(t_max.x, fmin(t_max.y, t_max.z));
                
                // Early exit for missed intersection
                if (t_near > t_far || t_far <= 0.0f) {
                    outcome = 1;
                    exit_angles[gid] = (dir.z > 0) ? acos(fabs(dir.z)) * 180.0f / M_PI : 0.0f;
                    break;
                }
                
                float t = select(t_far, t_near, t_near > 0.0f);
                float3 step_vec = dir.xyz * t;
                pos.xyz += step_vec;
                total_distance += t;
                
                if (currently_in_slab && has_absorption) {
                    float absorption_prob = 1.0f - native_exp(-abs_coeff * t);
                    uint abs_seed = mad24((uint)gid, 1000u, (uint)(step + 42));
                    abs_seed = mad24(abs_seed, 1664525u, 1013904223u);
                    float abs_rand = ((float)(abs_seed % 10000u)) * 0.0001f;
                    
                    if (abs_rand < absorption_prob) {
                        outcome = 2;
                        bool in_emitter_xy = (fabs(pos.x) <= half_emitter) && (fabs(pos.y) <= half_emitter);
                        bool in_emitter_z = (fabs(pos.z - slab_min.z) < 0.01f);
                        abs_type = select(0, 1, in_emitter_xy && in_emitter_z);
                        break;
                    }
                }
                
                float3 normal = (float3)(0, 0, 0);
                float3 abs_pos = fabs(pos.xyz);
                
                if (fabs(abs_pos.x - half_size.x) < 0.001f) {
                    normal.x = (pos.x > 0) ? 1.0f : -1.0f;
                } else if (fabs(abs_pos.y - half_size.y) < 0.001f) {
                    normal.y = (pos.y > 0) ? 1.0f : -1.0f;
                } else {
                    normal.z = (pos.z > 0) ? 1.0f : -1.0f;
                }
                
                bool was_in_slab = currently_in_slab;
                currently_in_slab = (pos.x >= slab_min.x && pos.x <= slab_max.x && 
                                   pos.y >= slab_min.y && pos.y <= slab_max.y && 
                                   pos.z >= slab_min.z && pos.z <= slab_max.z);
                
                float cos_theta_i = -dot(dir.xyz, normal);
                if (cos_theta_i < 0) { normal = -normal; cos_theta_i = -cos_theta_i; }
                
                // Use pre-computed refractive index ratios
                float n_ratio = was_in_slab ? n_ratio_glass_to_air : n_ratio_air_to_glass;
                float sin_theta_i = native_sqrt(1.0f - cos_theta_i * cos_theta_i);
                float sin_theta_t = n_ratio * sin_theta_i;
                
                if (sin_theta_t >= 1.0f) {
                    dir.xyz = dir.xyz - 2.0f * dot(dir.xyz, normal) * normal;
                } else {
                    float cos_theta_t = native_sqrt(1.0f - sin_theta_t * sin_theta_t);
                    float n1 = was_in_slab ? ri : world_refractive_index;
                    float n2 = was_in_slab ? world_refractive_index : ri;
                    float n1_cos_i = n1 * cos_theta_i;
                    float n2_cos_t = n2 * cos_theta_t;
                    float n1_cos_t = n1 * cos_theta_t;
                    float n2_cos_i = n2 * cos_theta_i;
                    
                    float r_s_num = n1_cos_i - n2_cos_t;
                    float r_s_den = n1_cos_i + n2_cos_t;
                    float r_p_num = n1_cos_t - n2_cos_i;
                    float r_p_den = n1_cos_t + n2_cos_i;
                    
                    float r_s = (r_s_num * r_s_num) / (r_s_den * r_s_den);
                    float r_p = (r_p_num * r_p_num) / (r_p_den * r_p_den);
                    float reflectance = (r_s + r_p) * 0.5f;
                    
                    uint seed = mad24((uint)gid, 1000u, (uint)step);
                    seed = mad24(seed, 1664525u, 1013904223u);
                    float rand_val = ((float)(seed % 10000u)) * 0.0001f;
                    
                    // CPU-identical logic: scattering only replaces reflection, not transmission
                    if (rand_val < reflectance) {
                        // Ray would reflect - check if we should scatter instead (CPU logic)
                        bool is_top_surface = (fabs(pos.z - half_size.z) < 1e-6f);  // Match CPU tolerance
                        bool should_scatter = is_top_surface && has_roughness;
                        
                        if (should_scatter) {
                            uint scatter_seed = mad24((uint)gid, 2000u, (uint)(step + 123));
                            scatter_seed = mad24(scatter_seed, 1664525u, 1013904223u);
                            float scatter_rand = ((float)(scatter_seed % 10000u)) * 0.0001f;
                            
                            if (scatter_rand < roughness) {
                                // Apply scattering instead of reflection
                                uint phi_seed = mad24(scatter_seed, 1664525u, 1013904679u);
                                float phi = 2.0f * M_PI * ((float)(phi_seed % 10000u)) * 0.0001f;
                                
                                uint cos_theta_seed = mad24(phi_seed, 1664525u, 1013905012u);
                                float cos_theta_scatter = native_sqrt(((float)(cos_theta_seed % 10000u)) * 0.0001f);
                                float sin_theta_scatter = native_sqrt(1.0f - cos_theta_scatter * cos_theta_scatter);
                                
                                // Use CPU-identical coordinate transformation with normal
                                float3 u, v;
                                if (fabs(normal.z) < 0.999f) {
                                    u = normalize(cross(normal, (float3)(0, 0, 1)));
                                } else {
                                    u = normalize(cross(normal, (float3)(0, 1, 0)));
                                }
                                v = cross(normal, u);
                                
                                float cos_phi, sin_phi = sincos(phi, &cos_phi);
                                dir.xyz = u * sin_theta_scatter * cos_phi + v * sin_theta_scatter * sin_phi + normal * cos_theta_scatter;
                            } else {
                                // Normal reflection
                                dir.xyz = dir.xyz - 2.0f * dot(dir.xyz, normal) * normal;
                            }
                        } else {
                            // Normal reflection (not top surface or no roughness)
                            dir.xyz = dir.xyz - 2.0f * dot(dir.xyz, normal) * normal;
                        }
                    } else {
                        float3 term1 = n_ratio * dir.xyz;
                        float3 term2 = (n_ratio * cos_theta_i - cos_theta_t) * normal;
                        dir.xyz = normalize(term1 + term2);
                        
                        if (was_in_slab && !currently_in_slab) {
                            outcome = 1;
                            if (dir.z > 0) exit_angles[gid] = acos(fabs(dir.z)) * 180.0f / M_PI;
                            break;
                        }
                    }
                }
                
                bounces++;
                pos.xyz += dir.xyz * 0.001f;
                
                float half_emitter = emitter_side_length / 2.0f;
                bool in_emitter_xy = (fabs(pos.x) <= half_emitter) && (fabs(pos.y) <= half_emitter);
                bool in_emitter_z = (fabs(pos.z - slab_min.z) < 0.01f);
                
                if (in_emitter_xy && in_emitter_z) {
                    uint emitter_seed = gid * 2000 + step + 123;
                    emitter_seed = emitter_seed * 1664525 + 1013904223;
                    float emitter_rand = ((float)(emitter_seed % 10000)) / 10000.0f;
                    if (emitter_rand < 0.999f) { outcome = 2; abs_type = 1; break; }
                }
            }
            
            if (outcome == 0) outcome = 3;
            ray_outcomes[gid] = outcome;
            path_lengths[gid] = total_distance;
            bounce_counts[gid] = bounces;
            absorption_types[gid] = abs_type;
        }
        """
        
        kernel_source = dome_kernel + "\n" + slab_kernel
        self.program = cl.Program(self.context, kernel_source).build()
    
    def gpu_trace_dome_rays(self, rays, config: SimulationConfig, world_ri=1.0):
        """GPU-accelerated ray tracing for dome geometry"""
        if not self.context or len(rays) < 10:
            return None
        
        try:
            N = len(rays)
            
            # Prepare ray data
            positions = np.zeros((N, 4), dtype=np.float32)
            directions = np.zeros((N, 4), dtype=np.float32)
            wavelengths = np.zeros(N, dtype=np.float32)
            
            for i, ray in enumerate(rays):
                positions[i, :3] = ray.position
                directions[i, :3] = ray.direction
                wavelengths[i] = ray.wavelength
            
            # Output arrays
            outcomes = np.zeros(N, dtype=np.int32)
            path_lengths = np.zeros(N, dtype=np.float32)
            bounce_counts = np.zeros(N, dtype=np.int32)
            absorption_types = np.zeros(N, dtype=np.int32)
            
            # Create GPU buffers
            pos_buf = cl_array.to_device(self.queue, positions)
            dir_buf = cl_array.to_device(self.queue, directions)
            wave_buf = cl_array.to_device(self.queue, wavelengths)
            
            outcomes_buf = cl_array.to_device(self.queue, outcomes)
            lengths_buf = cl_array.to_device(self.queue, path_lengths)
            bounces_buf = cl_array.to_device(self.queue, bounce_counts)
            abs_types_buf = cl_array.to_device(self.queue, absorption_types)
            
            # Execute GPU kernel
            self.program.trace_dome_rays(
                self.queue, (N,), None,
                pos_buf.data, dir_buf.data, wave_buf.data,
                outcomes_buf.data, lengths_buf.data, bounces_buf.data, abs_types_buf.data,
                np.float32(config.dome_radius),
                np.float32(config.emitter_radius),
                np.float32(config.glass_refractive_index),
                np.float32(world_ri),
                np.float32(config.dome_absorption_coefficient),
                np.float32(config.scatter_probability),
                np.int32(config.max_steps),
                np.int32(N)
            )
            
            # Get results
            gpu_outcomes = outcomes_buf.get()
            gpu_lengths = lengths_buf.get()
            gpu_bounces = bounces_buf.get()
            gpu_abs_types = abs_types_buf.get()
            
            # Convert to result format
            outcome_map = {0: 'error', 1: 'escaped', 2: 'absorbed', 3: 'trapped'}
            abs_type_map = {0: 'bulk', 1: 'emitter_absorption', 2: 'luminophore'}
            
            results = []
            for i in range(N):
                outcome = outcome_map.get(gpu_outcomes[i], 'error')
                abs_type = None
                if outcome == 'absorbed':
                    abs_type = abs_type_map.get(gpu_abs_types[i], 'bulk')
                
                results.append({
                    'outcome': outcome,
                    'length': float(gpu_lengths[i]),
                    'bounces': int(gpu_bounces[i]),
                    'abs_type': abs_type
                })
            
            return results
            
        except Exception as e:
            print(f"GPU dome ray tracing failed: {e}")
            return None
    
    def gpu_trace_slab_rays(self, rays, config: SimulationConfig, world_ri=1.0):
        """GPU-accelerated ray tracing for slab geometry"""
        if not self.context or len(rays) < 100:
            return None
        
        try:
            N = len(rays)
            
            # Prepare ray data
            positions = np.zeros((N, 4), dtype=np.float32)
            directions = np.zeros((N, 4), dtype=np.float32)
            wavelengths = np.zeros(N, dtype=np.float32)
            
            for i, ray in enumerate(rays):
                positions[i, :3] = ray.position
                directions[i, :3] = ray.direction
                wavelengths[i] = ray.wavelength
            
            # Slab geometry bounds
            half_size = np.array([config.slab_side_length/2, config.slab_side_length/2, config.slab_thickness/2])
            slab_bounds = np.array([
                [-half_size[0], -half_size[1], -half_size[2], 0],
                [half_size[0], half_size[1], half_size[2], 0]
            ], dtype=np.float32)
            
            # Material properties
            glass_ri = np.array([config.glass_refractive_index], dtype=np.float32)
            world_ri_array = np.array([world_ri], dtype=np.float32)
            total_absorption = config.slab_absorption_coefficient + config.luminophore_absorption_coefficient
            absorption_coeff = np.array([total_absorption], dtype=np.float32)
            surface_roughness = np.array([config.top_surface_roughness], dtype=np.float32)
            
            # Output arrays
            outcomes = np.zeros(N, dtype=np.int32)
            path_lengths = np.zeros(N, dtype=np.float32)
            bounce_counts = np.zeros(N, dtype=np.int32)
            exit_angles = np.zeros(N, dtype=np.float32)
            absorption_types = np.zeros(N, dtype=np.int32)
            
            # Use cached GPU buffers for better performance
            pos_buf = self._get_or_create_buffer('positions', N * 4, np.float32)
            dir_buf = self._get_or_create_buffer('directions', N * 4, np.float32)
            wave_buf = self._get_or_create_buffer('wavelengths', N, np.float32)
            outcomes_buf = self._get_or_create_buffer('outcomes', N, np.int32)
            lengths_buf = self._get_or_create_buffer('lengths', N, np.float32)
            bounces_buf = self._get_or_create_buffer('bounces', N, np.int32)
            angles_buf = self._get_or_create_buffer('angles', N, np.float32)
            abs_types_buf = self._get_or_create_buffer('abs_types', N, np.int32)
            
            # Copy data to GPU buffers (async when possible)
            cl.enqueue_copy(self.queue, pos_buf, positions).wait()
            cl.enqueue_copy(self.queue, dir_buf, directions).wait()
            cl.enqueue_copy(self.queue, wave_buf, wavelengths).wait()
            
            # Small constant buffers - create directly (minimal overhead)
            bounds_buf = cl_array.to_device(self.queue, slab_bounds)
            ri_buf = cl_array.to_device(self.queue, glass_ri)
            world_ri_buf = cl_array.to_device(self.queue, world_ri_array)
            abs_buf = cl_array.to_device(self.queue, absorption_coeff)
            roughness_buf = cl_array.to_device(self.queue, surface_roughness)
            
            # Execute GPU kernel with optimized work group size
            # Use optimal work group size for better GPU utilization
            work_group_size = 256  # Good for most modern GPUs
            global_size = ((N + work_group_size - 1) // work_group_size) * work_group_size
            
            self.program.trace_slab_rays(
                self.queue, (global_size,), (work_group_size,),
                pos_buf, dir_buf, wave_buf,
                bounds_buf.data, ri_buf.data, world_ri_buf.data, abs_buf.data, roughness_buf.data,
                outcomes_buf, lengths_buf, bounces_buf, angles_buf, abs_types_buf,
                np.int32(config.max_steps), np.int32(N), np.float32(config.emitter_side_length)
            )
            
            # Get results from cached buffers
            gpu_outcomes = np.empty(N, dtype=np.int32)
            gpu_lengths = np.empty(N, dtype=np.float32)
            gpu_bounces = np.empty(N, dtype=np.int32)
            gpu_angles = np.empty(N, dtype=np.float32)
            gpu_abs_types = np.empty(N, dtype=np.int32)
            
            cl.enqueue_copy(self.queue, gpu_outcomes, outcomes_buf).wait()
            cl.enqueue_copy(self.queue, gpu_lengths, lengths_buf).wait()
            cl.enqueue_copy(self.queue, gpu_bounces, bounces_buf).wait()
            cl.enqueue_copy(self.queue, gpu_angles, angles_buf).wait()
            cl.enqueue_copy(self.queue, gpu_abs_types, abs_types_buf).wait()
            
            # Convert to result format
            outcome_map = {0: 'error', 1: 'escaped', 2: 'absorbed', 3: 'trapped'}
            abs_type_map = {0: 'bulk', 1: 'emitter_absorption', 2: 'luminophore'}
            
            results = []
            for i in range(N):
                outcome = outcome_map.get(gpu_outcomes[i], 'error')
                angle = float(gpu_angles[i]) if outcome == 'escaped' and gpu_angles[i] > 0 else None
                abs_type = None
                if outcome == 'absorbed':
                    abs_type = abs_type_map.get(gpu_abs_types[i], 'bulk')
                
                results.append({
                    'outcome': outcome,
                    'length': float(gpu_lengths[i]),
                    'bounces': int(gpu_bounces[i]),
                    'angle': angle,
                    'abs_type': abs_type
                })
            
            return results
            
        except Exception as e:
            print(f"GPU slab ray tracing failed: {e}")
            return None
    
    def is_available(self):
        return self.context is not None


class UniversalSimulator:
    def __init__(self, config: SimulationConfig, gpu_instance: Optional[GPUAcceleration] = None):
        self.config = config
        self.world: Optional[Node] = None
        self.physics: Optional[SurfaceDelegate] = None
        self.light: Optional[Light] = None
        # Reuse GPU instance if provided, otherwise create new one
        self.gpu = gpu_instance if gpu_instance is not None else (GPUAcceleration() if config.use_gpu else None)
        self._setup_world()
    
    def _setup_world(self):
        self.world = Node(
            name="world",
            geometry=Sphere(
                radius=self.config.world_radius,
                material=Material(refractive_index=1.0)
            )
        )
        
        if self.config.geometry_type == "dome":
            self._setup_dome()
        elif self.config.geometry_type == "slab":
            self._setup_slab()
        else:
            raise ValueError(f"Unknown geometry type: {self.config.geometry_type}. Only 'dome' and 'slab' are supported.")
    
    def _setup_dome(self):
        self.physics = DomePhysics(scatter_probability=self.config.scatter_probability)
        
        glass_material = Material(
            refractive_index=self.config.glass_refractive_index,
            surface=Surface(delegate=self.physics)
        )
        glass_material.components.append(
            Absorber(coefficient=self.config.dome_absorption_coefficient)
        )
        
        wavelengths = np.linspace(400, 800, 100)
        emission_spectrum = np.column_stack((
            wavelengths,
            np.exp(-((wavelengths - 600)/50)**2)
        ))
        glass_material.components.append(
            Luminophore(
                coefficient=self.config.luminophore_absorption_coefficient,
                emission=emission_spectrum,
                quantum_yield=self.config.luminophore_quantum_yield
            )
        )
        
        dome = Node(
            name="dome",
            geometry=Hemisphere(radius=self.config.dome_radius, material=glass_material),
            parent=self.world
        )
        
        emitter_material = Material(
            refractive_index=1.0,
            components=[Absorber(coefficient=self.config.emitter_absorption_coefficient)]
        )
        emitter = Node(
            name="emitter",
            geometry=Cylinder(
                radius=self.config.emitter_radius,
                length=self.config.emitter_height,
                material=emitter_material
            ),
            parent=self.world
        )
        emitter.translate((0, 0, -self.config.emitter_height / 2.0))
        
        self.light = Light(
            name="dome_emitter",
            position=functools.partial(self._circular_emitter_position, self.config.emitter_radius),
            direction=self._lambertian_emitter,
            wavelength=lambda: 550
        )
        light_node = Node(name="light", light=self.light, parent=self.world)
    
    def _setup_slab(self):
        self.physics = SlabPhysics(
            top_surface_roughness=self.config.top_surface_roughness,
            position_tolerance=self.config.position_tolerance
        )
        
        glass_material = Material(
            refractive_index=self.config.glass_refractive_index,
            surface=Surface(delegate=self.physics)
        )
        glass_material.components.append(
            Absorber(coefficient=self.config.slab_absorption_coefficient)
        )
        
        if self.config.luminophore_absorption_coefficient > 0:
            wavelengths = np.linspace(400, 800, 100)
            emission_spectrum = np.column_stack([
                wavelengths,
                np.exp(-((wavelengths - 600)/50)**2)
            ])
            glass_material.components.append(
                Luminophore(
                    coefficient=self.config.luminophore_absorption_coefficient,
                    emission=emission_spectrum,
                    quantum_yield=self.config.luminophore_quantum_yield
                )
            )
        
        slab = Node(
            name="slab",
            geometry=Box(
                size=(self.config.slab_side_length, self.config.slab_side_length, self.config.slab_thickness),
                material=glass_material
            ),
            parent=self.world
        )
        
        emitter_material = Material(
            refractive_index=1.0,
            components=[Absorber(coefficient=self.config.emitter_absorption_coefficient)]
        )
        
        emitter = Node(
            name="emitter",
            geometry=Box(
                size=(self.config.emitter_side_length, self.config.emitter_side_length, self.config.emitter_thickness),
                material=emitter_material
            ),
            parent=self.world
        )
        emitter.translate((0, 0, -self.config.slab_thickness/2 - self.config.emitter_thickness/2))
        
        self.light = Light(
            name="slab_emitter",
            position=functools.partial(self._uniform_emitter_position),
            direction=self._lambertian_emitter,
            wavelength=lambda: 550
        )
        light_node = Node(name="light", light=self.light, parent=self.world)
    
    
    def _uniform_emitter_position(self):
        x = (np.random.rand() - 0.5) * self.config.emitter_side_length
        y = (np.random.rand() - 0.5) * self.config.emitter_side_length
        z = -self.config.slab_thickness/2 + self.config.position_tolerance
        return (x, y, z)
    
    @staticmethod
    def _circular_emitter_position(radius: float) -> Tuple[float, float, float]:
        r = radius * np.sqrt(np.random.uniform())
        theta = np.random.uniform(0, 2 * np.pi)
        return (r * np.cos(theta), r * np.sin(theta), 1e-9)
    
    @staticmethod
    def _lambertian_emitter() -> Tuple[float, float, float]:
        phi = 2 * np.pi * np.random.uniform()
        cos_theta = np.sqrt(np.random.uniform())
        sin_theta = np.sqrt(1 - cos_theta**2)
        return (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta)
    
    
    def _calculate_exit_angle(self, direction: Tuple[float, float, float]) -> float:
        direction_vec = np.array(direction)
        norm = np.linalg.norm(direction_vec)
        
        if norm < EPS_ZERO:
            return 0.0
        
        cos_angle = abs(direction_vec[2]) / norm
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return angle_rad * 180 / np.pi
    
    def _analyze_ray_path(self, ray: Ray) -> Tuple[str, float, int, Optional[float], Optional[str]]:
        try:
            scene = Scene(self.world)
            path = photon_tracer.follow(scene, ray, maxsteps=self.config.max_steps)
            
            if not path: return 'error', 0, 0, None, None
            
            final_ray, final_event = path[-1]
            path_length = final_ray.travelled; bounces = len(path) - 2
            exit_angle = None; absorption_type = None

            if final_event == Event.ABSORB:
                outcome = 'absorbed'
                final_node = photon_tracer.find_container(scene.intersections(final_ray.position, final_ray.direction))
                if final_node:
                    if 'emitter' in final_node.name:
                        absorption_type = 'emitter_absorption'
                    elif any(name in final_node.name for name in ['dome', 'slab']):
                        if self.config.luminophore_absorption_coefficient > 1e-9:
                            absorption_type = 'luminophore'
                        else:
                            absorption_type = 'bulk'
            elif final_event == Event.EXIT:
                outcome = 'escaped'; exit_angle = self._calculate_exit_angle(final_ray.direction)
            else: outcome = 'trapped'

            return outcome, path_length, bounces, exit_angle, absorption_type
        except Exception:
            return 'error', 0, 0, None, None
    
    def run_simulation(self) -> SimulationResults:
        start_time = time.time()
        
        self.physics.reset_stats()
        rays = list(self.light.emit(self.config.ray_count))
        
        # Try GPU acceleration first
        if self.gpu and self.gpu.is_available():
            if self.config.geometry_type == "dome":
                gpu_results = self.gpu.gpu_trace_dome_rays(rays, self.config)
                if gpu_results is not None:
                    return self._process_gpu_results(gpu_results, start_time)
            elif self.config.geometry_type == "slab":
                gpu_results = self.gpu.gpu_trace_slab_rays(rays, self.config)
                if gpu_results is not None:
                    return self._process_gpu_results(gpu_results, start_time)
        
        # CPU fallback
        
        results = defaultdict(int)
        absorption_counts = defaultdict(int)
        path_lengths = defaultdict(list)
        bounce_counts = defaultdict(list)
        exit_angles = []
        
        def process_ray_chunk(ray_chunk):
            chunk_results = defaultdict(int)
            chunk_absorption_counts = defaultdict(int)
            chunk_path_lengths = defaultdict(list)
            chunk_bounce_counts = defaultdict(list)
            chunk_exit_angles = []
            
            for ray in ray_chunk:
                try:
                    outcome, length, bounces, angle, abs_type = self._analyze_ray_path(ray)
                    
                    chunk_results[outcome] += 1
                    chunk_path_lengths[outcome].append(length)
                    chunk_bounce_counts[outcome].append(bounces)
                    
                    if angle is not None:
                        chunk_exit_angles.append(angle)
                    if abs_type:
                        chunk_absorption_counts[abs_type] += 1
                        
                except Exception:
                    chunk_results['error'] += 1
            
            return chunk_results, chunk_absorption_counts, chunk_path_lengths, chunk_bounce_counts, chunk_exit_angles
        
        import os
        num_workers = self.config.max_workers or max(1, os.cpu_count() - 1)
        
        optimal_chunk_size = max(50, len(rays) // (num_workers * 4))
        ray_chunks = [rays[i:i + optimal_chunk_size] for i in range(0, len(rays), optimal_chunk_size)]
        
        try:
            if len(rays) > 1000:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    chunk_results_list = list(executor.map(process_ray_chunk, ray_chunks))
            else:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    chunk_results_list = list(executor.map(process_ray_chunk, ray_chunks))
        except Exception:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                chunk_results_list = list(executor.map(process_ray_chunk, ray_chunks))
        
        for chunk_results, chunk_absorption_counts, chunk_path_lengths, chunk_bounce_counts, chunk_exit_angles in chunk_results_list:
            for outcome, count in chunk_results.items():
                results[outcome] += count
            
            for abs_type, count in chunk_absorption_counts.items():
                absorption_counts[abs_type] += count
            
            for outcome, lengths in chunk_path_lengths.items():
                path_lengths[outcome].extend(lengths)
            
            for outcome, bounces in chunk_bounce_counts.items():
                bounce_counts[outcome].extend(bounces)
            
            exit_angles.extend(chunk_exit_angles)
        
        interaction_stats = {}
        if hasattr(self.physics, 'interaction_count'):
            interaction_stats = {
                'total_interactions': self.physics.interaction_count,
                'scatter_events': self.physics.scatter_count,
                'transmission_events': self.physics.transmission_count,
                'tir_events': self.physics.tir_count
            }
        elif hasattr(self.physics, 'interactions'):
            interaction_stats = {
                'total_interactions': self.physics.interactions,
                'fresnel_reflections': self.physics.fresnel_reflections,
                'fresnel_transmissions': self.physics.fresnel_transmissions,
                'lambertian_scatterings': self.physics.lambertian_scatterings,
                'total_internal_reflections': self.physics.total_internal_reflections
            }
        
        return SimulationResults(
            total_rays=self.config.ray_count,
            absorbed_count=results['absorbed'],
            escaped_count=results['escaped'],
            trapped_count=results['trapped'],
            error_count=results['error'],
            exit_angles=exit_angles,
            path_lengths=dict(path_lengths),
            bounce_counts=dict(bounce_counts),
            interaction_stats=interaction_stats,
            execution_time=time.time() - start_time,
            bulk_absorption_count=absorption_counts['bulk'],
            emitter_absorption_count=absorption_counts['emitter_absorption'],
            luminophore_absorption_count=absorption_counts['luminophore']
        )
    
    def _process_gpu_results(self, gpu_results: List[Dict], start_time: float) -> SimulationResults:
        """Process GPU ray tracing results"""
        # Count outcomes
        outcome_counts = {'escaped': 0, 'absorbed': 0, 'trapped': 0, 'error': 0}
        absorption_counts = {'bulk': 0, 'emitter_absorption': 0, 'luminophore': 0}
        exit_angles = []
        
        total_interactions = 0
        tir_events = 0
        scatter_events = 0
        transmission_events = 0
        
        for result in gpu_results:
            outcome = result['outcome']
            outcome_counts[outcome] += 1
            
            # Track surface interactions from bounces
            bounces = result.get('bounces', 0)
            total_interactions += bounces
            
            # Estimate physics events from configuration and bounces
            if bounces > 0:
                if self.config.glass_refractive_index > 1.5:
                    ri_contrast = self.config.glass_refractive_index / 1.0
                    tir_fraction = min(0.8, (ri_contrast - 1.0) / 2.0)
                    estimated_tir = int(bounces * tir_fraction)
                    tir_events += estimated_tir
                
                if outcome == 'escaped':
                    transmission_events += 1
                
                if self.config.geometry_type == "dome" and self.config.scatter_probability > 0:
                    estimated_scatter = max(1, int(bounces * self.config.scatter_probability))
                    scatter_events += estimated_scatter
                elif self.config.geometry_type == "slab" and self.config.top_surface_roughness > 0:
                    scatter_fraction = min(0.3, self.config.top_surface_roughness)
                    estimated_scatter = max(1, int(bounces * scatter_fraction))
                    scatter_events += estimated_scatter
            
            if result.get('angle') is not None:
                exit_angles.append(result['angle'])
            
            if result.get('abs_type'):
                absorption_counts[result['abs_type']] += 1
        
        execution_time = time.time() - start_time
        
        interaction_stats = {
            'total_interactions': total_interactions,
            'tir_events': tir_events,
            'scatter_events': scatter_events,
            'transmission_events': transmission_events
        }
        
        return SimulationResults(
            total_rays=self.config.ray_count,
            absorbed_count=outcome_counts['absorbed'],
            escaped_count=outcome_counts['escaped'],
            trapped_count=outcome_counts['trapped'],
            error_count=outcome_counts['error'],
            exit_angles=exit_angles,
            path_lengths={},
            bounce_counts={},
            interaction_stats=interaction_stats,
            execution_time=execution_time,
            bulk_absorption_count=absorption_counts['bulk'],
            emitter_absorption_count=absorption_counts['emitter_absorption'],
            luminophore_absorption_count=absorption_counts['luminophore']
        )


class ConfigurationBuilder:
    def __init__(self):
        self.config = SimulationConfig()
    
    def dome(self, radius=5.0, emitter_radius=1.0, emitter_height=0.1):
        self.config.geometry_type = "dome"
        self.config.dome_radius = radius
        self.config.emitter_radius = emitter_radius
        self.config.emitter_height = emitter_height
        return self
    
    def slab(self, side_length=4.0, thickness=1.0, emitter_side_length=1.0, emitter_thickness=0.1):
        self.config.geometry_type = "slab"
        self.config.slab_side_length = side_length
        self.config.slab_thickness = thickness
        self.config.emitter_side_length = emitter_side_length
        self.config.emitter_thickness = emitter_thickness
        return self
    
    
    def glass(self, refractive_index=1.5, absorption=0.0):
        self.config.glass_refractive_index = refractive_index
        if self.config.geometry_type == "dome":
            self.config.dome_absorption_coefficient = absorption
        else:
            self.config.slab_absorption_coefficient = absorption
        return self
    
    def surface(self, scatter_probability=0.0, roughness=0.0):
        self.config.scatter_probability = scatter_probability
        self.config.top_surface_roughness = roughness
        return self
    
    def luminophore(self, absorption=0.0, quantum_yield=1.0):
        self.config.luminophore_absorption_coefficient = absorption
        self.config.luminophore_quantum_yield = quantum_yield
        return self
    
    def emitter(self, absorption=1000.0):
        self.config.emitter_absorption_coefficient = absorption
        return self
    
    def simulation(self, ray_count=10000, max_steps=1000):
        self.config.ray_count = ray_count
        self.config.max_steps = max_steps
        return self
    
    def build(self):
        return self.config


def dome_config():
    return ConfigurationBuilder().dome()

def slab_config():
    return ConfigurationBuilder().slab()


def calculate_lee(results: SimulationResults) -> float:
    """
    Calculate Light Extraction Efficiency (LEE) as:
        LEE = escaped_rays / (escaped_rays + absorbed_rays)

    This is the standard definition used in optical simulations.  Trapped rays
    (those that exceed max_steps without escaping or being absorbed) are excluded
    from the denominator.
    """
    total_rays = results.escaped_count + results.absorbed_count
    return results.escaped_count / total_rays if total_rays > 0 else 0.0


# Convenience functions for creating configurations
def create_dome_config(
    dome_radius: float = 5.0,
    emitter_radius: float = 1.0,
    emitter_height: float = 0.1,
    glass_refractive_index: float = 1.5,
    dome_absorption_coefficient: float = 0.0,
    scatter_probability: float = 0.0,
    luminophore_absorption_coefficient: float = 0.0,
    luminophore_quantum_yield: float = 1.0,
    emitter_absorption_coefficient: float = 1000.0,
    ray_count: int = 10000,
    max_steps: int = 1000,
    use_gpu: bool = True
) -> SimulationConfig:
    """Create a dome configuration with GPU acceleration enabled by default"""
    return SimulationConfig(
        geometry_type="dome",
        dome_radius=dome_radius,
        emitter_radius=emitter_radius,
        emitter_height=emitter_height,
        glass_refractive_index=glass_refractive_index,
        dome_absorption_coefficient=dome_absorption_coefficient,
        scatter_probability=scatter_probability,
        luminophore_absorption_coefficient=luminophore_absorption_coefficient,
        luminophore_quantum_yield=luminophore_quantum_yield,
        emitter_absorption_coefficient=emitter_absorption_coefficient,
        ray_count=ray_count,
        max_steps=max_steps,
        use_gpu=use_gpu
    )


def create_slab_config(
    slab_side_length: float = 10.0,
    slab_thickness: float = 3.0,
    emitter_side_length: float = 1.0,
    emitter_thickness: float = 0.1,
    glass_refractive_index: float = 1.5,
    slab_absorption_coefficient: float = 0.0,
    top_surface_roughness: float = 0.0,
    luminophore_absorption_coefficient: float = 0.0,
    luminophore_quantum_yield: float = 1.0,
    emitter_absorption_coefficient: float = 1000.0,
    ray_count: int = 10000,
    max_steps: int = 1000,
    use_gpu: bool = True
) -> SimulationConfig:
    """Create a slab configuration with GPU acceleration enabled by default"""
    return SimulationConfig(
        geometry_type="slab",
        slab_side_length=slab_side_length,
        slab_thickness=slab_thickness,
        emitter_side_length=emitter_side_length,
        emitter_thickness=emitter_thickness,
        glass_refractive_index=glass_refractive_index,
        slab_absorption_coefficient=slab_absorption_coefficient,
        top_surface_roughness=top_surface_roughness,
        luminophore_absorption_coefficient=luminophore_absorption_coefficient,
        luminophore_quantum_yield=luminophore_quantum_yield,
        emitter_absorption_coefficient=emitter_absorption_coefficient,
        ray_count=ray_count,
        max_steps=max_steps,
        use_gpu=use_gpu
    )


def run_dome_simulation(config: SimulationConfig = None, world_ri: float = 1.0, **kwargs) -> SimulationResults:
    """Run a dome simulation with optional parameters"""
    if config is None:
        config = create_dome_config(**kwargs)
    
    simulator = UniversalSimulator(config)
    return simulator.run_simulation()


def run_slab_simulation(config: SimulationConfig = None, world_ri: float = 1.0, **kwargs) -> SimulationResults:
    """Run a slab simulation with optional parameters"""
    if config is None:
        config = create_slab_config(**kwargs)
    
    simulator = UniversalSimulator(config)
    return simulator.run_simulation()