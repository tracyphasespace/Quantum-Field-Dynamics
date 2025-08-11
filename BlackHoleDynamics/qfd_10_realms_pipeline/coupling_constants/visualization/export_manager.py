"""
Export manager for coupling constants data and analysis results.

This module provides comprehensive export capabilities including JSON, YAML, CSV,
LaTeX tables, and publication-ready reports.
"""

import json
import yaml
import csv
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from pathlib import Path

from ..registry.parameter_registry import ParameterRegistry
from ..analysis.dependency_mapper import DependencyMapper
from ..analysis.sensitivity_analyzer import SensitivityAnalyzer
from ..analysis.realm_tracker import RealmTracker


class ExportManager:
    """
    Manager for exporting coupling constants data and analysis results.
    
    Provides multi-format export capabilities including structured data export,
    publication-ready tables, and comprehensive analysis reports.
    """
    
    def __init__(self, registry: ParameterRegistry):
        """
        Initialize export manager.
        
        Args:
            registry: ParameterRegistry instance to export from
        """
        self.registry = registry
        
    def export_parameters_json(self, output_path: str, 
                             include_history: bool = True,
                             include_constraints: bool = True) -> None:
        """
        Export parameters to JSON format.
        
        Args:
            output_path: Path to save JSON file
            include_history: Whether to include parameter change history
            include_constraints: Whether to include constraint information
        """
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_parameters": len(self.registry.get_all_parameters()),
                "export_type": "parameters_json"
            },
            "parameters": {}
        }
        
        for param_name, param in self.registry.get_all_parameters().items():
            param_data = {
                "name": param_name,
                "value": param.value,
                "uncertainty": param.uncertainty,
                "fixed_by_realm": param.fixed_by_realm,
                "last_modified": param.last_modified.isoformat(),
                "metadata": param.metadata
            }
            
            if include_constraints:
                param_data["constraints"] = [
                    {
                        "realm": c.realm,
                        "type": c.constraint_type.value,
                        "min_value": c.min_value,
                        "max_value": c.max_value,
                        "target_value": c.target_value,
                        "tolerance": c.tolerance,
                        "notes": c.notes,
                        "active": c.active,
                        "created_at": c.created_at.isoformat()
                    }
                    for c in param.constraints
                ]
            
            if include_history:
                param_data["history"] = [
                    {
                        "timestamp": change.timestamp.isoformat(),
                        "realm": change.realm,
                        "old_value": change.old_value,
                        "new_value": change.new_value,
                        "reason": change.reason
                    }
                    for change in param.history
                ]
            
            export_data["parameters"][param_name] = param_data
        
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Parameters exported to JSON: {output_path}")
    
    def export_parameters_yaml(self, output_path: str) -> None:
        """
        Export parameters to YAML format.
        
        Args:
            output_path: Path to save YAML file
        """
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_parameters": len(self.registry.get_all_parameters())
            },
            "parameters": {}
        }
        
        for param_name, param in self.registry.get_all_parameters().items():
            param_data = {
                "value": param.value,
                "fixed_by_realm": param.fixed_by_realm,
                "constraints": len(param.constraints),
                "last_modified": param.last_modified.isoformat()
            }
            
            if param.metadata:
                param_data["metadata"] = param.metadata
            
            export_data["parameters"][param_name] = param_data
        
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(export_data, f, default_flow_style=False, indent=2)
        
        print(f"Parameters exported to YAML: {output_path}")
    
    def export_parameters_csv(self, output_path: str) -> None:
        """
        Export parameters to CSV format.
        
        Args:
            output_path: Path to save CSV file
        """
        rows = []
        
        for param_name, param in self.registry.get_all_parameters().items():
            row = {
                "parameter_name": param_name,
                "value": param.value,
                "uncertainty": param.uncertainty,
                "fixed_by_realm": param.fixed_by_realm,
                "constraint_count": len(param.constraints),
                "change_count": len(param.history),
                "last_modified": param.last_modified.isoformat(),
                "unit": param.metadata.get("unit", ""),
                "note": param.metadata.get("note", "")
            }
            
            # Add constraint summary
            constraint_types = [c.constraint_type.value for c in param.constraints]
            row["constraint_types"] = "; ".join(set(constraint_types))
            
            # Add bounds if available
            min_vals = [c.min_value for c in param.constraints if c.min_value is not None]
            max_vals = [c.max_value for c in param.constraints if c.max_value is not None]
            
            row["min_bound"] = min(min_vals) if min_vals else None
            row["max_bound"] = max(max_vals) if max_vals else None
            
            rows.append(row)
        
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write CSV
        if rows:
            fieldnames = rows[0].keys()
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        print(f"Parameters exported to CSV: {output_path}")
    
    def export_latex_table(self, output_path: str, 
                          parameters: Optional[List[str]] = None,
                          include_constraints: bool = True,
                          table_caption: str = "QFD Coupling Constants") -> None:
        """
        Export parameters as LaTeX table.
        
        Args:
            output_path: Path to save LaTeX file
            parameters: List of parameters to include (all if None)
            include_constraints: Whether to include constraint information
            table_caption: Caption for the LaTeX table
        """
        all_params = self.registry.get_all_parameters()
        
        if parameters:
            params_to_export = {name: param for name, param in all_params.items() 
                              if name in parameters}
        else:
            # Export parameters with values
            params_to_export = {name: param for name, param in all_params.items() 
                              if param.value is not None}
        
        if not params_to_export:
            print("No parameters to export to LaTeX")
            return
        
        latex_content = []
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        
        if include_constraints:
            latex_content.append("\\begin{tabular}{|l|c|c|l|l|}")
            latex_content.append("\\hline")
            latex_content.append("Parameter & Value & Uncertainty & Constraints & Realm \\\\")
        else:
            latex_content.append("\\begin{tabular}{|l|c|c|l|}")
            latex_content.append("\\hline")
            latex_content.append("Parameter & Value & Uncertainty & Realm \\\\")
        
        latex_content.append("\\hline")
        
        for param_name, param in params_to_export.items():
            # Format parameter name (escape underscores)
            latex_name = param_name.replace("_", "\\_")
            
            # Format value
            if param.value is not None:
                if abs(param.value) < 1e-3 or abs(param.value) > 1e3:
                    value_str = f"{param.value:.2e}"
                else:
                    value_str = f"{param.value:.6f}"
            else:
                value_str = "---"
            
            # Format uncertainty
            if param.uncertainty is not None:
                if abs(param.uncertainty) < 1e-3 or abs(param.uncertainty) > 1e3:
                    uncertainty_str = f"{param.uncertainty:.2e}"
                else:
                    uncertainty_str = f"{param.uncertainty:.6f}"
            else:
                uncertainty_str = "---"
            
            # Format realm
            realm_str = param.fixed_by_realm or "---"
            realm_str = realm_str.replace("_", "\\_")
            
            if include_constraints:
                # Format constraints
                constraint_strs = []
                for constraint in param.constraints:
                    if constraint.constraint_type.value == "bounded":
                        if constraint.min_value is not None and constraint.max_value is not None:
                            constraint_strs.append(f"[{constraint.min_value:.2e}, {constraint.max_value:.2e}]")
                        elif constraint.min_value is not None:
                            constraint_strs.append(f"≥{constraint.min_value:.2e}")
                        elif constraint.max_value is not None:
                            constraint_strs.append(f"≤{constraint.max_value:.2e}")
                    elif constraint.constraint_type.value == "fixed":
                        constraint_strs.append(f"={constraint.target_value:.2e}")
                    elif constraint.constraint_type.value == "target":
                        tol_str = f"±{constraint.tolerance:.2e}" if constraint.tolerance else ""
                        constraint_strs.append(f"≈{constraint.target_value:.2e}{tol_str}")
                
                constraints_str = "; ".join(constraint_strs) if constraint_strs else "---"
                
                latex_content.append(f"{latex_name} & {value_str} & {uncertainty_str} & {constraints_str} & {realm_str} \\\\")
            else:
                latex_content.append(f"{latex_name} & {value_str} & {uncertainty_str} & {realm_str} \\\\")
        
        latex_content.append("\\hline")
        latex_content.append("\\end{tabular}")
        latex_content.append(f"\\caption{{{table_caption}}}")
        latex_content.append("\\label{tab:coupling_constants}")
        latex_content.append("\\end{table}")
        
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(latex_content))
        
        print(f"LaTeX table exported: {output_path}")
    
    def create_comprehensive_report(self, output_dir: str,
                                  dependency_mapper: Optional[DependencyMapper] = None,
                                  sensitivity_analyzer: Optional[SensitivityAnalyzer] = None,
                                  realm_tracker: Optional[RealmTracker] = None) -> None:
        """
        Create comprehensive analysis report with multiple formats.
        
        Args:
            output_dir: Directory to save all report files
            dependency_mapper: Optional DependencyMapper for dependency analysis
            sensitivity_analyzer: Optional SensitivityAnalyzer for sensitivity data
            realm_tracker: Optional RealmTracker for execution data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating comprehensive report in {output_dir}/...")
        
        # 1. Export parameters in multiple formats
        self.export_parameters_json(f"{output_dir}/parameters.json")
        self.export_parameters_yaml(f"{output_dir}/parameters.yaml")
        self.export_parameters_csv(f"{output_dir}/parameters.csv")
        
        # 2. Export LaTeX table
        self.export_latex_table(f"{output_dir}/parameters_table.tex")
        
        # 3. Export dependency analysis (if available)
        if dependency_mapper:
            dependency_mapper.export_graph('json', f"{output_dir}/dependency_graph.json")
            
            # Export dependency summary
            stats = dependency_mapper.get_summary_statistics()
            with open(f"{output_dir}/dependency_summary.json", 'w') as f:
                json.dump(stats, f, indent=2)
        
        # 4. Export sensitivity analysis (if available)
        if sensitivity_analyzer:
            sensitivity_analyzer.export_results(f"{output_dir}/sensitivity_analysis.json")
        
        # 5. Export realm execution data (if available)
        if realm_tracker:
            realm_tracker.export_execution_log(f"{output_dir}/realm_execution.json")
        
        # 6. Create summary report
        self._create_summary_report(output_dir, dependency_mapper, 
                                  sensitivity_analyzer, realm_tracker)
        
        print(f"Comprehensive report created in {output_dir}/")
    
    def export_for_publication(self, output_dir: str,
                             key_parameters: Optional[List[str]] = None,
                             paper_title: str = "QFD Coupling Constants Analysis") -> None:
        """
        Export data in publication-ready formats.
        
        Args:
            output_dir: Directory to save publication files
            key_parameters: List of key parameters for publication
            paper_title: Title for the publication materials
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating publication-ready exports in {output_dir}/...")
        
        # 1. High-quality LaTeX table
        if key_parameters:
            self.export_latex_table(f"{output_dir}/main_results_table.tex",
                                   parameters=key_parameters,
                                   table_caption=f"{paper_title}: Key Parameters")
        
        # 2. Supplementary data in CSV
        self.export_parameters_csv(f"{output_dir}/supplementary_data.csv")
        
        # 3. Machine-readable JSON for reproducibility
        self.export_parameters_json(f"{output_dir}/complete_dataset.json",
                                   include_history=True, include_constraints=True)
        
        # 4. Create README for publication data
        self._create_publication_readme(output_dir, paper_title, key_parameters)
        
        print(f"Publication materials created in {output_dir}/")
    
    def _create_summary_report(self, output_dir: str,
                             dependency_mapper: Optional[DependencyMapper],
                             sensitivity_analyzer: Optional[SensitivityAnalyzer],
                             realm_tracker: Optional[RealmTracker]) -> None:
        """Create summary report in markdown format."""
        
        report_lines = []
        report_lines.append(f"# QFD Coupling Constants Analysis Report")
        report_lines.append(f"")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"")
        
        # Parameter summary
        all_params = self.registry.get_all_parameters()
        params_with_values = {name: param for name, param in all_params.items() 
                            if param.value is not None}
        
        report_lines.append(f"## Parameter Summary")
        report_lines.append(f"")
        report_lines.append(f"- Total parameters: {len(all_params)}")
        report_lines.append(f"- Parameters with values: {len(params_with_values)}")
        report_lines.append(f"- Total constraints: {sum(len(p.constraints) for p in all_params.values())}")
        report_lines.append(f"")
        
        # Key parameters table
        if params_with_values:
            report_lines.append(f"## Key Parameters")
            report_lines.append(f"")
            report_lines.append(f"| Parameter | Value | Realm | Constraints |")
            report_lines.append(f"|-----------|-------|-------|-------------|")
            
            for param_name, param in list(params_with_values.items())[:10]:  # Top 10
                value_str = f"{param.value:.6e}" if param.value else "---"
                realm_str = param.fixed_by_realm or "---"
                constraint_count = len(param.constraints)
                report_lines.append(f"| {param_name} | {value_str} | {realm_str} | {constraint_count} |")
            
            report_lines.append(f"")
        
        # Dependency analysis summary
        if dependency_mapper:
            stats = dependency_mapper.get_summary_statistics()
            report_lines.append(f"## Dependency Analysis")
            report_lines.append(f"")
            report_lines.append(f"- Total dependencies: {stats.get('total_dependencies', 0)}")
            report_lines.append(f"- Parameter clusters: {stats.get('total_clusters', 0)}")
            report_lines.append(f"- Graph density: {stats.get('graph_density', 0):.3f}")
            report_lines.append(f"- Most influential parameter: {stats.get('most_influential_parameter', 'N/A')}")
            report_lines.append(f"")
        
        # Sensitivity analysis summary
        if sensitivity_analyzer and sensitivity_analyzer.sensitivity_results:
            report_lines.append(f"## Sensitivity Analysis")
            report_lines.append(f"")
            report_lines.append(f"- Sensitivity analyses performed: {len(sensitivity_analyzer.sensitivity_results)}")
            report_lines.append(f"- Monte Carlo analyses: {len(sensitivity_analyzer.monte_carlo_results)}")
            report_lines.append(f"- Parameter rankings: {len(sensitivity_analyzer.parameter_rankings)}")
            report_lines.append(f"")
        
        # Realm execution summary
        if realm_tracker:
            summary = realm_tracker.get_execution_summary()
            if summary.get('total_executions', 0) > 0:
                report_lines.append(f"## Realm Execution Summary")
                report_lines.append(f"")
                report_lines.append(f"- Total executions: {summary.get('total_executions', 0)}")
                report_lines.append(f"- Successful executions: {summary.get('successful_executions', 0)}")
                report_lines.append(f"- Failed executions: {summary.get('failed_executions', 0)}")
                report_lines.append(f"- Total execution time: {summary.get('total_execution_time_ms', 0):.2f} ms")
                report_lines.append(f"- Parameters modified: {summary.get('total_parameters_modified', 0)}")
                report_lines.append(f"")
        
        # Files included
        report_lines.append(f"## Files Included")
        report_lines.append(f"")
        report_lines.append(f"- `parameters.json`: Complete parameter data with history and constraints")
        report_lines.append(f"- `parameters.yaml`: Parameter summary in YAML format")
        report_lines.append(f"- `parameters.csv`: Parameter data in CSV format for analysis")
        report_lines.append(f"- `parameters_table.tex`: LaTeX table for publication")
        
        if dependency_mapper:
            report_lines.append(f"- `dependency_graph.json`: Parameter dependency graph data")
            report_lines.append(f"- `dependency_summary.json`: Dependency analysis statistics")
        
        if sensitivity_analyzer:
            report_lines.append(f"- `sensitivity_analysis.json`: Complete sensitivity analysis results")
        
        if realm_tracker:
            report_lines.append(f"- `realm_execution.json`: Realm execution history and statistics")
        
        # Write report
        with open(f"{output_dir}/README.md", 'w') as f:
            f.write('\n'.join(report_lines))
    
    def _create_publication_readme(self, output_dir: str, paper_title: str,
                                 key_parameters: Optional[List[str]]) -> None:
        """Create README for publication data."""
        
        readme_lines = []
        readme_lines.append(f"# {paper_title} - Supplementary Data")
        readme_lines.append(f"")
        readme_lines.append(f"This directory contains the supplementary data and analysis results for:")
        readme_lines.append(f"**{paper_title}**")
        readme_lines.append(f"")
        readme_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        readme_lines.append(f"")
        
        readme_lines.append(f"## Files Description")
        readme_lines.append(f"")
        readme_lines.append(f"### Main Results")
        readme_lines.append(f"- `main_results_table.tex`: LaTeX table with key parameters for inclusion in paper")
        readme_lines.append(f"- `supplementary_data.csv`: Complete parameter dataset in CSV format")
        readme_lines.append(f"")
        
        readme_lines.append(f"### Complete Dataset")
        readme_lines.append(f"- `complete_dataset.json`: Machine-readable complete dataset with:")
        readme_lines.append(f"  - All parameter values and uncertainties")
        readme_lines.append(f"  - Complete constraint information")
        readme_lines.append(f"  - Parameter change history")
        readme_lines.append(f"  - Metadata and provenance information")
        readme_lines.append(f"")
        
        if key_parameters:
            readme_lines.append(f"## Key Parameters")
            readme_lines.append(f"")
            readme_lines.append(f"The following parameters are highlighted as key results:")
            for param in key_parameters:
                readme_lines.append(f"- `{param}`")
            readme_lines.append(f"")
        
        readme_lines.append(f"## Data Format")
        readme_lines.append(f"")
        readme_lines.append(f"### JSON Structure")
        readme_lines.append(f"The complete dataset JSON file contains:")
        readme_lines.append(f"```json")
        readme_lines.append(f'{{')
        readme_lines.append(f'  "metadata": {{ ... }},')
        readme_lines.append(f'  "parameters": {{')
        readme_lines.append(f'    "parameter_name": {{')
        readme_lines.append(f'      "value": float,')
        readme_lines.append(f'      "uncertainty": float,')
        readme_lines.append(f'      "constraints": [...],')
        readme_lines.append(f'      "history": [...]')
        readme_lines.append(f'    }}')
        readme_lines.append(f'  }}')
        readme_lines.append(f'}}')
        readme_lines.append(f"```")
        readme_lines.append(f"")
        
        readme_lines.append(f"## Reproducibility")
        readme_lines.append(f"")
        readme_lines.append(f"This data was generated using the QFD Coupling Constants Analysis Framework.")
        readme_lines.append(f"The complete analysis can be reproduced using the provided dataset and")
        readme_lines.append(f"the framework code available at [repository URL].")
        readme_lines.append(f"")
        
        readme_lines.append(f"## Citation")
        readme_lines.append(f"")
        readme_lines.append(f"If you use this data in your research, please cite:")
        readme_lines.append(f"[Citation information to be added]")
        
        with open(f"{output_dir}/README.md", 'w') as f:
            f.write('\n'.join(readme_lines))