#!/usr/bin/env python3
"""Slab parameter sweep demonstration"""

import numpy as np
from simulation_physics import create_slab_config
from parameter_sweep import GPUParameterSweep, SweepPlotter

def main():
    print("Slab Parameter Sweep Demo")
    print("=" * 30)
    
    sweep = GPUParameterSweep()
    plotter = SweepPlotter()
    
    # 1. Thickness sweep
    print("\n1. Thickness sweep:")
    thicknesses = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    param_values, lee_values = sweep.sweep_1d('slab_thickness', thicknesses)
    plotter.plot_1d(param_values, lee_values, 'slab_thickness', save_path='slab_thickness.png')
    
    opt_idx = np.argmax(lee_values)
    print(f"   Best: thickness={param_values[opt_idx]:.1f}, LEE={lee_values[opt_idx]:.3f}")
    
    # 2. Surface roughness sweep
    print("\n2. Surface roughness sweep:")
    roughness_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    param_values, lee_values = sweep.sweep_1d('top_surface_roughness', roughness_values)
    plotter.plot_1d(param_values, lee_values, 'top_surface_roughness', save_path='slab_roughness.png')
    
    opt_idx = np.argmax(lee_values)
    print(f"   Best: roughness={param_values[opt_idx]:.1f}, LEE={lee_values[opt_idx]:.3f}")
    
    # 3. Refractive index comparison
    print("\n3. RI comparison:")
    ri_values = [1.5, 2.0, 2.5]
    data_sets = {}
    
    for ri in ri_values:
        config = create_slab_config(glass_refractive_index=ri, ray_count=2000)
        ri_sweep = GPUParameterSweep(config)
        _, lee_values = ri_sweep.sweep_1d('slab_thickness', [1.0, 2.0, 3.0, 4.0])
        data_sets[f'RI={ri}'] = ([1.0, 2.0, 3.0, 4.0], lee_values)
    
    plotter.plot_comparison(data_sets, 'slab_thickness', save_path='slab_ri_comparison.png')
    print("   Comparison plot saved")
    
    # 4. 2D optimization
    print("\n4. 2D optimization:")
    t_values = [1.0, 2.0, 3.0, 4.0]
    ri_values = [1.5, 2.0, 2.5]
    
    t_grid, ri_grid, lee_grid = sweep.sweep_2d('slab_thickness', t_values, 
                                               'glass_refractive_index', ri_values)
    plotter.plot_2d(t_grid, ri_grid, lee_grid, 'slab_thickness', 'glass_refractive_index',
                   save_path='slab_2d_optimization.png')
    
    max_idx = np.unravel_index(np.argmax(lee_grid), lee_grid.shape)
    opt_thickness = t_grid[max_idx]
    opt_ri = ri_grid[max_idx]
    opt_lee = lee_grid[max_idx]
    print(f"   Global optimum: thickness={opt_thickness:.1f}, RI={opt_ri:.1f}, LEE={opt_lee:.3f}")
    
    print(f"\nFiles created:")
    print("  slab_thickness.png")
    print("  slab_roughness.png") 
    print("  slab_ri_comparison.png")
    print("  slab_2d_optimization.png")

if __name__ == "__main__":
    main()