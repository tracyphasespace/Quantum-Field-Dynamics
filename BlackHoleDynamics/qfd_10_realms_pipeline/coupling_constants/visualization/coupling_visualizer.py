"""
Coupling constants visualizer for dependency graphs and constraint plots.

This module provides comprehensive visualization capabilities for coupling
constants, including dependency graphs, constraint plots, and parameter
relationship visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
from datetime import datetime

from ..registry.parameter_registry import ParameterRegistry
from ..analysis.dependency_mapper import DependencyMapper
from ..analysis.sensitivity_analyzer import SensitivityAnalyzer


class CouplingVisualizer:
    """
    Visualizer for coupling constants dependency graphs and constraint plots.
    
    Provides comprehensive visualization capabilities including dependency graphs,
    parameter constraint plots, sensitivity analysis plots, and realm execution flows.
    """
    
    def __init__(self, registry: ParameterRegistry):
        """
        Initialize coupling visualizer.
        
        Args:
            registry: ParameterRegistry instance to visualize
        """
        self.registry = registry
        self.figure_size = (12, 8)
        self.dpi = 300
        self.color_scheme = {
            'nodes': {
                'fixed': '#FF6B6B',      # Red for fixed parameters
                'bounded': '#4ECDC4',    # Teal for bounded parameters
                'target': '#45B7D1',     # Blue for target parameters
                'unconstrained': '#96CEB4'  # Green for unconstrained
            },
            'edges': {
                'constraint': '#2C3E50',     # Dark blue for constraint dependencies
                'realm_order': '#8E44AD',    # Purple for realm order dependencies
                'physics': '#E74C3C',        # Red for physics dependencies
                'derived': '#F39C12'         # Orange for derived dependencies
            },
            'realms': {
                'cmb': '#E74C3C',           # Red
                'cosmic': '#3498DB',        # Blue
                'scales': '#2ECC71',        # Green
                'em': '#F39C12',            # Orange
                'particles': '#9B59B6',     # Purple
                'config': '#34495E'         # Dark gray
            }
        }
    
    def plot_dependency_graph(self, dependency_mapper: DependencyMapper, 
                            output_path: str, layout: str = 'spring',
                            show_edge_labels: bool = True,
                            highlight_critical_path: bool = True) -> None:
        """
        Plot the parameter dependency graph.
        
        Args:
            dependency_mapper: DependencyMapper with built dependency graph
            output_path: Path to save the plot
            layout: Graph layout algorithm ('spring', 'circular', 'hierarchical')
            show_edge_labels: Whether to show edge labels
            highlight_critical_path: Whether to highlight the critical path
        """
        graph = dependency_mapper.dependency_graph
        
        if not graph.nodes():
            print("No dependency graph to plot")
            return
        
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=3, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'hierarchical':
            pos = self._hierarchical_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        # Color nodes by constraint type
        node_colors = []
        node_sizes = []
        
        for node in graph.nodes():
            param = self.registry.get_parameter(node)
            if param:
                # Determine node color based on constraints
                if param.is_fixed():
                    color = self.color_scheme['nodes']['fixed']
                elif any(c.constraint_type.value == 'bounded' for c in param.get_active_constraints()):
                    color = self.color_scheme['nodes']['bounded']
                elif any(c.constraint_type.value == 'target' for c in param.get_active_constraints()):
                    color = self.color_scheme['nodes']['target']
                else:
                    color = self.color_scheme['nodes']['unconstrained']
                
                # Size based on number of constraints
                size = 300 + len(param.constraints) * 100
            else:
                color = self.color_scheme['nodes']['unconstrained']
                size = 300
            
            node_colors.append(color)
            node_sizes.append(size)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8)
        
        # Draw edges by type
        edge_types = {}
        for source, target, data in graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            if edge_type not in edge_types:
                edge_types[edge_type] = []
            edge_types[edge_type].append((source, target))
        
        for edge_type, edges in edge_types.items():
            color = self.color_scheme['edges'].get(edge_type, '#666666')
            nx.draw_networkx_edges(graph, pos, edgelist=edges, 
                                 edge_color=color, alpha=0.6, width=1.5)
        
        # Highlight critical path if requested
        if highlight_critical_path:
            critical_path = dependency_mapper.find_critical_path()
            if len(critical_path) > 1:
                critical_edges = [(critical_path[i], critical_path[i+1]) 
                                for i in range(len(critical_path)-1)
                                if graph.has_edge(critical_path[i], critical_path[i+1])]
                if critical_edges:
                    nx.draw_networkx_edges(graph, pos, edgelist=critical_edges,
                                         edge_color='red', width=3, alpha=0.8)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold')
        
        # Add edge labels if requested
        if show_edge_labels and len(graph.edges()) < 50:  # Only for smaller graphs
            edge_labels = {}
            for source, target, data in graph.edges(data=True):
                weight = data.get('weight', 0)
                if weight > 0.1:  # Only show significant weights
                    edge_labels[(source, target)] = f"{weight:.2f}"
            
            if edge_labels:
                nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=6)
        
        # Create legend
        self._add_dependency_legend()
        
        plt.title("Coupling Constants Dependency Graph", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Dependency graph saved to {output_path}")
    
    def plot_parameter_constraints(self, output_path: str, 
                                 parameters: Optional[List[str]] = None) -> None:
        """
        Plot parameter constraints and current values.
        
        Args:
            output_path: Path to save the plot
            parameters: List of parameters to plot (all if None)
        """
        all_params = self.registry.get_all_parameters()
        
        if parameters:
            params_to_plot = {name: param for name, param in all_params.items() 
                            if name in parameters}
        else:
            # Plot parameters with constraints and values
            params_to_plot = {name: param for name, param in all_params.items() 
                            if param.constraints and param.value is not None}
        
        if not params_to_plot:
            print("No parameters with constraints and values to plot")
            return
        
        n_params = len(params_to_plot)
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 2*n_params), dpi=self.dpi)
        
        if n_params == 1:
            axes = [axes]
        
        for i, (param_name, param) in enumerate(params_to_plot.items()):
            ax = axes[i]
            
            # Get constraint bounds
            min_vals = []
            max_vals = []
            target_vals = []
            
            for constraint in param.get_active_constraints():
                if constraint.min_value is not None:
                    min_vals.append(constraint.min_value)
                if constraint.max_value is not None:
                    max_vals.append(constraint.max_value)
                if constraint.target_value is not None:
                    target_vals.append(constraint.target_value)
            
            # Determine plot range
            all_vals = [param.value] + min_vals + max_vals + target_vals
            all_vals = [v for v in all_vals if v is not None]
            
            if not all_vals:
                continue
            
            plot_min = min(all_vals) * 0.9
            plot_max = max(all_vals) * 1.1
            
            if plot_min == plot_max:
                plot_min -= 1
                plot_max += 1
            
            # Plot constraint regions
            y_pos = 0.5
            
            # Bounded constraints
            for constraint in param.get_active_constraints():
                if constraint.constraint_type.value == 'bounded':
                    min_val = constraint.min_value if constraint.min_value is not None else plot_min
                    max_val = constraint.max_value if constraint.max_value is not None else plot_max
                    
                    rect = patches.Rectangle((min_val, y_pos-0.1), max_val-min_val, 0.2,
                                           linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.3)
                    ax.add_patch(rect)
            
            # Target constraints
            for constraint in param.get_active_constraints():
                if constraint.constraint_type.value == 'target' and constraint.target_value is not None:
                    tolerance = constraint.tolerance or 0
                    target = constraint.target_value
                    
                    if tolerance > 0:
                        rect = patches.Rectangle((target-tolerance, y_pos-0.05), 2*tolerance, 0.1,
                                               linewidth=1, edgecolor='green', facecolor='lightgreen', alpha=0.5)
                        ax.add_patch(rect)
                    
                    ax.axvline(target, color='green', linestyle='--', alpha=0.7, label='Target')
            
            # Fixed constraints
            for constraint in param.get_active_constraints():
                if constraint.constraint_type.value == 'fixed' and constraint.target_value is not None:
                    ax.axvline(constraint.target_value, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Fixed')
            
            # Current value
            if param.value is not None:
                ax.axvline(param.value, color='black', linestyle='-', linewidth=3, alpha=0.9, label='Current')
            
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(0, 1)
            ax.set_ylabel(param_name, fontweight='bold')
            ax.set_yticks([])
            
            # Add constraint info
            constraint_info = []
            for constraint in param.get_active_constraints():
                constraint_info.append(f"{constraint.constraint_type.value} ({constraint.realm})")
            
            if constraint_info:
                ax.text(0.02, 0.8, '\n'.join(constraint_info), transform=ax.transAxes,
                       fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle("Parameter Constraints and Current Values", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Parameter constraints plot saved to {output_path}")
    
    def plot_realm_execution_flow(self, execution_history: List[Any], 
                                output_path: str) -> None:
        """
        Plot realm execution flow and timing.
        
        Args:
            execution_history: List of RealmExecutionResult objects
            output_path: Path to save the plot
        """
        if not execution_history:
            print("No execution history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, dpi=self.dpi)
        
        # Plot 1: Execution timeline
        realm_names = [result.realm_name for result in execution_history]
        execution_times = [result.execution_time_ms for result in execution_history]
        statuses = [result.status.value for result in execution_history]
        
        # Color by status
        colors = []
        for status in statuses:
            if status == 'completed':
                colors.append('green')
            elif status == 'failed':
                colors.append('red')
            else:
                colors.append('orange')
        
        bars = ax1.bar(range(len(realm_names)), execution_times, color=colors, alpha=0.7)
        ax1.set_xlabel('Realm Execution Order')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('Realm Execution Timeline')
        ax1.set_xticks(range(len(realm_names)))
        ax1.set_xticklabels(realm_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, time_val in zip(bars, execution_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{time_val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Parameters modified per realm
        param_counts = [len(result.parameters_modified) for result in execution_history]
        
        bars2 = ax2.bar(range(len(realm_names)), param_counts, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Realm')
        ax2.set_ylabel('Parameters Modified')
        ax2.set_title('Parameters Modified by Each Realm')
        ax2.set_xticks(range(len(realm_names)))
        ax2.set_xticklabels(realm_names, rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars2, param_counts):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{count}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Realm execution flow plot saved to {output_path}")
    
    def plot_parameter_evolution(self, parameter_names: List[str], 
                               output_path: str) -> None:
        """
        Plot parameter value evolution over time.
        
        Args:
            parameter_names: List of parameter names to plot
            output_path: Path to save the plot
        """
        fig, axes = plt.subplots(len(parameter_names), 1, 
                               figsize=(12, 3*len(parameter_names)), dpi=self.dpi)
        
        if len(parameter_names) == 1:
            axes = [axes]
        
        for i, param_name in enumerate(parameter_names):
            param = self.registry.get_parameter(param_name)
            if not param or not param.history:
                axes[i].text(0.5, 0.5, f"No history for {param_name}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(param_name)
                continue
            
            # Extract history
            times = list(range(len(param.history)))
            values = [change.new_value for change in param.history]
            realms = [change.realm for change in param.history]
            
            # Plot evolution
            axes[i].plot(times, values, 'o-', linewidth=2, markersize=6)
            
            # Color points by realm
            unique_realms = list(set(realms))
            realm_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_realms)))
            realm_color_map = dict(zip(unique_realms, realm_colors))
            
            for j, (time, value, realm) in enumerate(zip(times, values, realms)):
                axes[i].scatter(time, value, c=[realm_color_map[realm]], s=50, alpha=0.8)
            
            axes[i].set_xlabel('Change Number')
            axes[i].set_ylabel('Parameter Value')
            axes[i].set_title(f'{param_name} Evolution')
            axes[i].grid(True, alpha=0.3)
            
            # Add realm labels
            for j, (time, value, realm) in enumerate(zip(times, values, realms)):
                if j % 2 == 0:  # Label every other point to avoid crowding
                    axes[i].annotate(realm, (time, value), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=8, alpha=0.7)
        
        plt.suptitle("Parameter Evolution Over Time", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Parameter evolution plot saved to {output_path}")
    
    def create_dashboard(self, dependency_mapper: DependencyMapper,
                        sensitivity_analyzer: Optional[SensitivityAnalyzer] = None,
                        execution_history: Optional[List[Any]] = None,
                        output_dir: str = "coupling_dashboard") -> None:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            dependency_mapper: DependencyMapper with built dependency graph
            sensitivity_analyzer: Optional SensitivityAnalyzer with results
            execution_history: Optional realm execution history
            output_dir: Directory to save dashboard plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating coupling constants dashboard in {output_dir}/...")
        
        # 1. Dependency graph
        self.plot_dependency_graph(dependency_mapper, 
                                 f"{output_dir}/dependency_graph.png")
        
        # 2. Parameter constraints
        self.plot_parameter_constraints(f"{output_dir}/parameter_constraints.png")
        
        # 3. Realm execution flow (if available)
        if execution_history:
            self.plot_realm_execution_flow(execution_history, 
                                         f"{output_dir}/realm_execution_flow.png")
        
        # 4. Parameter evolution for key parameters
        key_params = ["PPN_gamma", "PPN_beta", "k_J", "xi", "psi_s0"]
        existing_params = [p for p in key_params if self.registry.get_parameter(p)]
        if existing_params:
            self.plot_parameter_evolution(existing_params, 
                                        f"{output_dir}/parameter_evolution.png")
        
        # 5. Sensitivity plots (if available)
        if sensitivity_analyzer and sensitivity_analyzer.sensitivity_results:
            sensitivity_analyzer.generate_sensitivity_plots(f"{output_dir}/sensitivity")
        
        # 6. Create summary HTML
        self._create_dashboard_html(output_dir, dependency_mapper, 
                                  sensitivity_analyzer, execution_history)
        
        print(f"Dashboard created successfully in {output_dir}/")
    
    def _hierarchical_layout(self, graph) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout based on realm dependencies."""
        # Simple hierarchical layout - could be improved
        try:
            levels = {}
            for node in graph.nodes():
                # Assign level based on in-degree (parameters with more dependencies go lower)
                in_degree = graph.in_degree(node)
                levels[node] = in_degree
            
            max_level = max(levels.values()) if levels else 0
            pos = {}
            level_counts = {}
            
            for node, level in levels.items():
                if level not in level_counts:
                    level_counts[level] = 0
                
                x = level_counts[level]
                y = max_level - level
                pos[node] = (x, y)
                level_counts[level] += 1
            
            return pos
        except:
            # Fallback to spring layout
            return nx.spring_layout(graph)
    
    def _add_dependency_legend(self) -> None:
        """Add legend for dependency graph."""
        legend_elements = []
        
        # Node types
        for node_type, color in self.color_scheme['nodes'].items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10,
                                            label=f'{node_type.title()} Parameter'))
        
        # Edge types
        for edge_type, color in self.color_scheme['edges'].items():
            legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2,
                                            label=f'{edge_type.title()} Dependency'))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    def _create_dashboard_html(self, output_dir: str, dependency_mapper: DependencyMapper,
                             sensitivity_analyzer: Optional[SensitivityAnalyzer],
                             execution_history: Optional[List[Any]]) -> None:
        """Create HTML dashboard summary."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>QFD Coupling Constants Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .plot {{ text-align: center; margin: 20px 0; }}
        .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        .stats {{ display: flex; justify-content: space-around; }}
        .stat-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>QFD Coupling Constants Analysis Dashboard</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>System Overview</h2>
        <div class="stats">
            <div class="stat-box">
                <h3>{len(self.registry.get_all_parameters())}</h3>
                <p>Total Parameters</p>
            </div>
            <div class="stat-box">
                <h3>{len(dependency_mapper.dependencies)}</h3>
                <p>Dependencies</p>
            </div>
            <div class="stat-box">
                <h3>{len(dependency_mapper.clusters)}</h3>
                <p>Parameter Clusters</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Dependency Graph</h2>
        <div class="plot">
            <img src="dependency_graph.png" alt="Dependency Graph">
            <p>Parameter dependency relationships across QFD realms</p>
        </div>
    </div>
    
    <div class="section">
        <h2>Parameter Constraints</h2>
        <div class="plot">
            <img src="parameter_constraints.png" alt="Parameter Constraints">
            <p>Current parameter values and their constraint boundaries</p>
        </div>
    </div>
"""
        
        if execution_history:
            html_content += """
    <div class="section">
        <h2>Realm Execution</h2>
        <div class="plot">
            <img src="realm_execution_flow.png" alt="Realm Execution Flow">
            <p>Realm execution timeline and parameter modifications</p>
        </div>
    </div>
"""
        
        html_content += """
    <div class="section">
        <h2>Parameter Evolution</h2>
        <div class="plot">
            <img src="parameter_evolution.png" alt="Parameter Evolution">
            <p>Evolution of key parameters over time</p>
        </div>
    </div>
"""
        
        if sensitivity_analyzer:
            html_content += """
    <div class="section">
        <h2>Sensitivity Analysis</h2>
        <p>Detailed sensitivity analysis plots available in the sensitivity/ subdirectory</p>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(f"{output_dir}/dashboard.html", "w") as f:
            f.write(html_content)
        
        print(f"Dashboard HTML created at {output_dir}/dashboard.html")