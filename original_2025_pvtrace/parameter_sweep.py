#!/usr/bin/env python3
"""Parameter sweep framework for GPU ray tracing"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation_physics import (
    SimulationConfig, SimulationResults, UniversalSimulator,
    create_slab_config, create_dome_config, calculate_lee
)


class GPUParameterSweep:
    def __init__(self, base_config: Optional[SimulationConfig] = None):
        if base_config is None:
            # Default to slab configuration
            base_config = create_slab_config(
                slab_side_length=10.0,
                slab_thickness=2.0, 
                emitter_side_length=1.0,
                glass_refractive_index=1.6,
                ray_count=1000,
                use_gpu=True
            )
        self.base_config = base_config
        self._gpu_instance = None  # Reuse GPU instance for better performance
    
    def run_single(self, **param_overrides) -> Tuple[float, SimulationResults]:
        try:
            # Create modified config
            config_dict = asdict(self.base_config)
            config_dict.update(param_overrides)
            config = SimulationConfig(**config_dict)
            
            # Reuse GPU instance if available for better performance
            if config.use_gpu and self._gpu_instance is None:
                from simulation_physics import GPUAcceleration
                self._gpu_instance = GPUAcceleration()
            
            # Run simulation with shared GPU instance
            simulator = UniversalSimulator(config, gpu_instance=self._gpu_instance)
            results = simulator.run_simulation()
            lee = calculate_lee(results)
            
            return lee, results
            
        except Exception as e:
            print(f"Simulation failed: {e}")
            return 0.0, None
    
    def sweep_1d(self, param_name: str, values: List[float], 
                 parallel: bool = True, max_workers: int = 4, save_csv: str = None) -> Tuple[List[float], List[float]]:
        start_time = time.time()
        
        def run_config(value):
            return self.run_single(**{param_name: value})
        
        if parallel and len(values) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(run_config, values))
        else:
            results = [run_config(value) for value in values]
        
        lee_values = [result[0] for result in results]
        
        if save_csv:
            save_results(values, results, param_name, save_csv)
        
        return values, lee_values
    
    def sweep_2d(self, param1_name: str, param1_values: List[float],
                 param2_name: str, param2_values: List[float],
                 parallel: bool = True, max_workers: int = 4, save_csv: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sweep two parameters to create a 2D grid
        
        Args:
            param1_name: First parameter name
            param1_values: First parameter values
            param2_name: Second parameter name  
            param2_values: Second parameter values
            parallel: Whether to use parallel processing
            max_workers: Maximum number of parallel workers
            save_csv: Optional CSV filename to save detailed results
            
        Returns:
            (param1_grid, param2_grid, lee_grid)
        """
        start_time = time.time()
        
        # Create parameter combinations
        param_combinations = []
        for p1 in param1_values:
            for p2 in param2_values:
                param_combinations.append({param1_name: p1, param2_name: p2})
        
        def run_config(params):
            return self.run_single(**params)
        
        # Run simulations
        if parallel and len(param_combinations) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(run_config, param_combinations))
        else:
            results = [run_config(params) for params in param_combinations]
        
        # Reshape results into 2D grid
        lee_values = [result[0] for result in results]
        lee_grid = np.array(lee_values).reshape(len(param1_values), len(param2_values))
        
        # Save CSV if requested
        if save_csv:
            # Flatten parameter combinations for CSV
            param1_flat = [params[param1_name] for params in param_combinations]
            param2_flat = [params[param2_name] for params in param_combinations]
            
            # Create combined data structure
            csv_data = {param1_name: param1_flat, param2_name: param2_flat}
            for i, result in enumerate(results):
                lee, sim_results = result
                if sim_results is not None:
                    csv_data.setdefault('LEE', []).append(lee)
                    csv_data.setdefault('Total_Rays', []).append(sim_results.total_rays)
                    csv_data.setdefault('Escaped_Count', []).append(sim_results.escaped_count)
                    csv_data.setdefault('Absorbed_Count', []).append(sim_results.absorbed_count)
                    csv_data.setdefault('Trapped_Count', []).append(sim_results.trapped_count)
                    csv_data.setdefault('Error_Count', []).append(sim_results.error_count)
                    csv_data.setdefault('Escaped_Percentage', []).append(sim_results.escaped_count / sim_results.total_rays * 100)
                    csv_data.setdefault('Absorbed_Percentage', []).append(sim_results.absorbed_count / sim_results.total_rays * 100)
                else:
                    for key in ['LEE', 'Total_Rays', 'Escaped_Count', 'Absorbed_Count', 'Trapped_Count', 'Error_Count', 'Escaped_Percentage', 'Absorbed_Percentage']:
                        csv_data.setdefault(key, []).append(0)
            
            df = pd.DataFrame(csv_data)
            df.to_csv(save_csv, index=False)
        
        # Create meshgrid
        param2_grid, param1_grid = np.meshgrid(param2_values, param1_values)
        
        return param1_grid, param2_grid, lee_grid
    
    def sweep_custom(self, parameter_sets: List[Dict[str, float]], 
                     parallel: bool = True, max_workers: int = 4, save_csv: str = None) -> List[Tuple[Dict, float]]:
        """
        Sweep custom parameter combinations
        
        Args:
            parameter_sets: List of parameter dictionaries to test
            parallel: Whether to use parallel processing
            max_workers: Maximum number of parallel workers
            save_csv: Optional CSV filename to save detailed results
            
        Returns:
            List of (parameters, LEE) tuples
        """
        start_time = time.time()
        
        def run_config(params):
            lee, sim_results = self.run_single(**params)
            return params, lee, sim_results
        
        if parallel and len(parameter_sets) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(run_config, parameter_sets))
        else:
            results = [run_config(params) for params in parameter_sets]
        
        if save_csv:
            # Create flattened data structure for CSV
            csv_data = {}
            param_names = set()
            for params, _, _ in results:
                param_names.update(params.keys())
            
            # Initialize parameter columns
            for param_name in sorted(param_names):
                csv_data[param_name] = []
            
            # Fill data
            for params, lee, sim_results in results:
                # Add parameter values
                for param_name in sorted(param_names):
                    csv_data[param_name].append(params.get(param_name, 0))
                
                # Add simulation results
                if sim_results is not None:
                    csv_data.setdefault('LEE', []).append(lee)
                    csv_data.setdefault('Total_Rays', []).append(sim_results.total_rays)
                    csv_data.setdefault('Escaped_Count', []).append(sim_results.escaped_count)
                    csv_data.setdefault('Absorbed_Count', []).append(sim_results.absorbed_count)
                    csv_data.setdefault('Trapped_Count', []).append(sim_results.trapped_count)
                    csv_data.setdefault('Error_Count', []).append(sim_results.error_count)
                    csv_data.setdefault('Escaped_Percentage', []).append(sim_results.escaped_count / sim_results.total_rays * 100)
                    csv_data.setdefault('Absorbed_Percentage', []).append(sim_results.absorbed_count / sim_results.total_rays * 100)
                else:
                    for key in ['LEE', 'Total_Rays', 'Escaped_Count', 'Absorbed_Count', 'Trapped_Count', 'Error_Count', 'Escaped_Percentage', 'Absorbed_Percentage']:
                        csv_data.setdefault(key, []).append(0)
            
            df = pd.DataFrame(csv_data)
            df.to_csv(save_csv, index=False)
        
        # Return original format for compatibility
        return [(params, lee) for params, lee, _ in results]


class SweepPlotter:
    def __init__(self):
        plt.rcParams.update({
            'font.size': 12,
            'axes.linewidth': 1.2,
            'figure.dpi': 100,
            'savefig.dpi': 300
        })
    
    def _format_parameter_label(self, param_name: str) -> str:
        """Format parameter name for axis labels"""
        label_map = {
            # Slab parameters
            'slab_thickness': 'Slab Thickness (mm)',
            'slab_side_length': 'Slab Width (mm)', 
            'emitter_side_length': 'Emitter Width (mm)',
            'top_surface_roughness': 'Surface Roughness',
            # Dome parameters
            'dome_radius': 'Dome Radius (mm)',
            'emitter_radius': 'Emitter Radius (mm)',
            'emitter_height': 'Emitter Height (mm)',
            # Common parameters
            'glass_refractive_index': 'Refractive Index',
            'ray_count': 'Number of Rays'
        }
        return label_map.get(param_name, param_name.replace('_', ' ').title())
    
    def plot_1d(self, param_values: List[float], lee_values: List[float], 
                param_name: str, title: str = None, save_path: str = None):
        """Plot 1D parameter sweep results"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Main plot
        ax.plot(param_values, lee_values, 'o-', linewidth=2, markersize=6,
               color='#1f77b4', markerfacecolor='white', markeredgewidth=1.5)
        
        # Labels and formatting
        ax.set_xlabel(self._format_parameter_label(param_name))
        ax.set_ylabel('Light Extraction Efficiency')
        ax.set_title(title or f'LEE vs {self._format_parameter_label(param_name)}')
        ax.grid(True, alpha=0.3)
        
        # Mark optimal point
        max_idx = np.argmax(lee_values)
        opt_x, opt_y = param_values[max_idx], lee_values[max_idx]
        ax.plot(opt_x, opt_y, 'rs', markersize=8, 
               label=f'Optimum: {opt_x:.3f}, LEE = {opt_y:.4f}')
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_2d(self, param1_grid: np.ndarray, param2_grid: np.ndarray, 
                lee_grid: np.ndarray, param1_name: str, param2_name: str,
                title: str = None, save_path: str = None):
        """Plot 2D parameter sweep heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        contour = ax.contourf(param1_grid, param2_grid, lee_grid, 
                             levels=20, cmap='viridis')
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label('Light Extraction Efficiency', rotation=270, labelpad=20)
        
        # Contour lines
        lines = ax.contour(param1_grid, param2_grid, lee_grid, 
                          levels=10, colors='white', alpha=0.5, linewidths=0.8)
        ax.clabel(lines, inline=True, fontsize=10, fmt='%.3f')
        
        # Labels
        ax.set_xlabel(self._format_parameter_label(param1_name))
        ax.set_ylabel(self._format_parameter_label(param2_name))
        ax.set_title(title or f'LEE vs {self._format_parameter_label(param1_name)} & {self._format_parameter_label(param2_name)}')
        
        # Mark optimal point
        max_idx = np.unravel_index(np.argmax(lee_grid), lee_grid.shape)
        opt_param1 = param1_grid[max_idx]
        opt_param2 = param2_grid[max_idx]
        opt_lee = lee_grid[max_idx]
        
        ax.plot(opt_param1, opt_param2, 'r*', markersize=15,
               label=f'Optimum: ({opt_param1:.2f}, {opt_param2:.2f}), LEE = {opt_lee:.4f}')
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_comparison(self, data_sets: Dict[str, Tuple[List, List]], 
                       param_name: str, title: str = None, save_path: str = None):
        """Plot comparison of multiple 1D sweeps"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (label, (param_values, lee_values)) in enumerate(data_sets.items()):
            color = colors[i % len(colors)]
            ax.plot(param_values, lee_values, 'o-', linewidth=2, markersize=4,
                   color=color, label=label)
        
        ax.set_xlabel(self._format_parameter_label(param_name))
        ax.set_ylabel('Light Extraction Efficiency')
        ax.set_title(title or f'LEE Comparison vs {self._format_parameter_label(param_name)}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()


def save_results(param_values: List[float], results_data: List[Tuple], 
                 param_name: str, filename: str):
    """Save sweep results with detailed ray statistics to CSV file"""
    data = {param_name: param_values}
    
    for result in results_data:
        lee, sim_results = result
        if sim_results is not None:
            data.setdefault('LEE', []).append(lee)
            data.setdefault('Total_Rays', []).append(sim_results.total_rays)
            data.setdefault('Escaped_Count', []).append(sim_results.escaped_count)
            data.setdefault('Absorbed_Count', []).append(sim_results.absorbed_count) 
            data.setdefault('Trapped_Count', []).append(sim_results.trapped_count)
            data.setdefault('Error_Count', []).append(sim_results.error_count)
            data.setdefault('Escaped_Percentage', []).append(sim_results.escaped_count / sim_results.total_rays * 100)
            data.setdefault('Absorbed_Percentage', []).append(sim_results.absorbed_count / sim_results.total_rays * 100)
        else:
            for key in ['LEE', 'Total_Rays', 'Escaped_Count', 'Absorbed_Count', 'Trapped_Count', 'Error_Count', 'Escaped_Percentage', 'Absorbed_Percentage']:
                data.setdefault(key, []).append(0)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


# Example usage functions
def example_thickness_sweep():
    """Example: Sweep slab thickness with detailed CSV output"""
    print("=== Slab Thickness Sweep Example ===")
    
    sweep = GPUParameterSweep()
    thicknesses = np.linspace(0.5, 3.0, 8)
    
    param_values, lee_values = sweep.sweep_1d('slab_thickness', thicknesses.tolist(), 
                                             save_csv='thickness_sweep_results.csv')
    
    plotter = SweepPlotter()
    plotter.plot_1d(param_values, lee_values, 'slab_thickness', 
                   save_path='thickness_sweep.png')
    
    print("Detailed ray statistics saved to thickness_sweep_results.csv")
    return param_values, lee_values


def example_ri_comparison():
    """Example: Compare different refractive indices"""
    print("=== Refractive Index Comparison Example ===")
    
    thicknesses = np.linspace(0.5, 3.0, 6)
    ri_values = [1.4, 1.8, 2.2]
    
    data_sets = {}
    
    for ri in ri_values:
        base_config = create_slab_config(glass_refractive_index=ri, ray_count=800)
        sweep = GPUParameterSweep(base_config)
        
        _, lee_values = sweep.sweep_1d('slab_thickness', thicknesses.tolist())
        data_sets[f'RI = {ri}'] = (thicknesses.tolist(), lee_values)
    
    plotter = SweepPlotter()
    plotter.plot_comparison(data_sets, 'slab_thickness', 
                           title='LEE vs Thickness for Different Refractive Indices',
                           save_path='ri_comparison.png')
    
    return data_sets


def example_2d_optimization():
    """Example: 2D optimization sweep with detailed CSV output"""
    print("=== 2D Optimization Example ===")
    
    sweep = GPUParameterSweep()
    
    thicknesses = np.linspace(0.5, 2.5, 4)
    ri_values = np.linspace(1.4, 2.2, 4)
    
    t_grid, ri_grid, lee_grid = sweep.sweep_2d(
        'slab_thickness', thicknesses.tolist(),
        'glass_refractive_index', ri_values.tolist(),
        max_workers=2,  # Limit parallel workers
        save_csv='2d_optimization_results.csv'
    )
    
    plotter = SweepPlotter()
    plotter.plot_2d(t_grid, ri_grid, lee_grid, 
                   'slab_thickness', 'glass_refractive_index',
                   save_path='optimization_2d.png')
    
    print("Detailed ray statistics saved to 2d_optimization_results.csv")
    return t_grid, ri_grid, lee_grid


def example_dome_sweep():
    """Example: Sweep dome radius"""
    print("=== Dome Radius Sweep Example ===")
    
    # Create dome configuration
    base_config = create_dome_config(
        dome_radius=5.0,
        emitter_radius=1.0,
        glass_refractive_index=1.6,
        ray_count=1000,
        use_gpu=True
    )
    
    sweep = GPUParameterSweep(base_config)
    radii = np.linspace(3.0, 8.0, 6)
    
    param_values, lee_values = sweep.sweep_1d('dome_radius', radii.tolist())
    
    plotter = SweepPlotter()
    plotter.plot_1d(param_values, lee_values, 'dome_radius', 
                   save_path='dome_sweep.png')
    
    return param_values, lee_values


def example_custom_sweep():
    """Example: Custom parameter combinations with detailed CSV output"""
    print("=== Custom Parameter Sweep Example ===")
    
    sweep = GPUParameterSweep()
    
    # Define custom parameter combinations to test
    parameter_sets = [
        {'slab_thickness': 1.0, 'glass_refractive_index': 1.5, 'top_surface_roughness': 0.0},
        {'slab_thickness': 1.0, 'glass_refractive_index': 1.5, 'top_surface_roughness': 0.3},
        {'slab_thickness': 2.0, 'glass_refractive_index': 2.0, 'top_surface_roughness': 0.0},
        {'slab_thickness': 2.0, 'glass_refractive_index': 2.0, 'top_surface_roughness': 0.3},
    ]
    
    results = sweep.sweep_custom(parameter_sets, save_csv='custom_sweep_results.csv')
    
    print("Custom sweep results:")
    for params, lee in results:
        print(f"  {params} -> LEE = {lee:.4f}")
    
    print("Detailed ray statistics saved to custom_sweep_results.csv")
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        example_name = sys.argv[1].lower()
        
        if example_name == 'thickness':
            example_thickness_sweep()
        elif example_name == 'ri':
            example_ri_comparison()
        elif example_name == '2d':
            example_2d_optimization()
        elif example_name == 'dome':
            example_dome_sweep()
        elif example_name == 'custom':
            example_custom_sweep()
        else:
            print(f"Unknown example: {example_name}")
            print("Available: thickness, ri, 2d, dome, custom")
    else:
        print("Flexible Parameter Sweep Framework")
        print("Usage: python parameter_sweep.py [example_name]")
        print("Available examples: thickness, ri, 2d, dome, custom")
        print()
        print("Quick test:")
        example_thickness_sweep()