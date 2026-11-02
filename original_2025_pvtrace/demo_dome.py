#!/usr/bin/env python3
"""Dome parameter sweep demonstration"""

import numpy as np
from simulation_physics import create_dome_config
from parameter_sweep import GPUParameterSweep, SweepPlotter

def main():
    print("Dome Parameter Sweep Demo")
    print("=" * 30)
    
    # Base dome configuration
    base_config = create_dome_config(dome_radius=5.0, emitter_radius=1.0, 
                                    glass_refractive_index=1.8, ray_count=3000, use_gpu=True)
    sweep = GPUParameterSweep(base_config)
    plotter = SweepPlotter()
    
    # 1. Radius sweep
    print("\n1. Radius sweep:")
    radii = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    param_values, lee_values = sweep.sweep_1d('dome_radius', radii)
    plotter.plot_1d(param_values, lee_values, 'dome_radius', save_path='dome_radius.png')
    
    opt_idx = np.argmax(lee_values)
    print(f"   Best: radius={param_values[opt_idx]:.1f}, LEE={lee_values[opt_idx]:.3f}")
    
    # 2. Scatter probability sweep
    print("\n2. Scatter probability sweep:")
    scatter_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    param_values, lee_values = sweep.sweep_1d('scatter_probability', scatter_values)
    plotter.plot_1d(param_values, lee_values, 'scatter_probability', save_path='dome_scatter.png')
    
    opt_idx = np.argmax(lee_values)
    print(f"   Best: scatter={param_values[opt_idx]:.1f}, LEE={lee_values[opt_idx]:.3f}")
    
    # 3. Refractive index comparison
    print("\n3. RI comparison:")
    ri_values = [1.5, 2.0, 2.5]
    data_sets = {}
    
    for ri in ri_values:
        config = create_dome_config(glass_refractive_index=ri, ray_count=2000, use_gpu=True)
        ri_sweep = GPUParameterSweep(config)
        _, lee_values = ri_sweep.sweep_1d('dome_radius', [3.0, 4.0, 5.0, 6.0])
        data_sets[f'RI={ri}'] = ([3.0, 4.0, 5.0, 6.0], lee_values)
    
    plotter.plot_comparison(data_sets, 'dome_radius', save_path='dome_ri_comparison.png')
    print("   Comparison plot saved")
    
    # 4. 2D optimization
    print("\n4. 2D optimization:")
    r_values = [3.0, 4.0, 5.0, 6.0]
    ri_values = [1.5, 2.0, 2.5]
    
    r_grid, ri_grid, lee_grid = sweep.sweep_2d('dome_radius', r_values,
                                               'glass_refractive_index', ri_values)
    plotter.plot_2d(r_grid, ri_grid, lee_grid, 'dome_radius', 'glass_refractive_index',
                   save_path='dome_2d_optimization.png')
    
    max_idx = np.unravel_index(np.argmax(lee_grid), lee_grid.shape)
    opt_radius = r_grid[max_idx]
    opt_ri = ri_grid[max_idx]
    opt_lee = lee_grid[max_idx]
    print(f"   Global optimum: radius={opt_radius:.1f}, RI={opt_ri:.1f}, LEE={opt_lee:.3f}")
    
    print(f"\nFiles created:")
    print("  dome_radius.png")
    print("  dome_scatter.png")
    print("  dome_ri_comparison.png") 
    print("  dome_2d_optimization.png")

if __name__ == "__main__":
    main()