#!/usr/bin/env python3
"""
Dependency analysis demonstration.

This script demonstrates the dependency mapping and analysis capabilities
for coupling constants in the QFD framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from coupling_constants.registry.parameter_registry import ParameterRegistry
from coupling_constants.config.yaml_loader import load_parameters_from_yaml
from coupling_constants.analysis.dependency_mapper import DependencyMapper
import json


def main():
    print("=== QFD Coupling Constants Dependency Analysis Demo ===\n")
    
    # 1. Load configuration and set up registry
    print("1. Loading QFD configuration and setting up parameter registry...")
    registry = ParameterRegistry()
    load_parameters_from_yaml("qfd_params/defaults.yaml", registry)
    
    print(f"   ✓ Loaded {len(registry.get_all_parameters())} parameters")
    
    # 2. Simulate realm execution with parameter updates
    print("\n2. Simulating realm execution with parameter updates...")
    
    # CMB realm
    registry.update_parameter("T_CMB_K", 2.725, "cmb_config", "CMB temperature")
    registry.update_parameter("k_J", 1e-12, "realm0_cmb", "Vacuum drag")
    registry.update_parameter("psi_s0", -1.5, "realm0_cmb", "Thermalization zeropoint")
    
    # Scales realm
    registry.update_parameter("PPN_gamma", 1.000001, "realm3_scales", "PPN gamma")
    registry.update_parameter("PPN_beta", 0.99999, "realm3_scales", "PPN beta")
    registry.update_parameter("E0", 1e3, "realm3_scales", "Energy scale")
    registry.update_parameter("L0", 1e-10, "realm3_scales", "Length scale")
    
    # EM realm
    registry.update_parameter("xi", 2.0, "realm4_em", "EM response")
    registry.update_parameter("k_c2", 0.5, "realm4_em", "EM coupling")
    registry.update_parameter("k_EM", 1.5, "realm4_em", "EM field coupling")
    
    print(f"   ✓ Updated 10 parameters across multiple realms")
    
    # 3. Create dependency mapper and build graph
    print("\n3. Building dependency graph and analyzing relationships...")
    mapper = DependencyMapper(registry)
    graph = mapper.build_dependency_graph()
    
    print(f"   ✓ Built dependency graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    print(f"   ✓ Identified {len(mapper.dependencies)} parameter dependencies")
    
    # 4. Analyze dependency types
    print("\n4. Dependency analysis by type:")
    dep_types = {}
    for dep in mapper.dependencies:
        dep_types[dep.dependency_type] = dep_types.get(dep.dependency_type, 0) + 1
    
    for dep_type, count in dep_types.items():
        print(f"   • {dep_type}: {count} dependencies")
    
    # Show some example dependencies
    print("\n   Example dependencies:")
    for i, dep in enumerate(mapper.dependencies[:5]):
        print(f"   {i+1}. {dep.source_param} → {dep.target_param}")
        print(f"      Type: {dep.dependency_type}, Strength: {dep.strength:.3f}")
        print(f"      Realm: {dep.realm}, Description: {dep.description}")
    
    # 5. Find critical path
    print("\n5. Critical path analysis:")
    critical_path = mapper.find_critical_path()
    
    if critical_path:
        print(f"   Critical path ({len(critical_path)} parameters):")
        for i, param in enumerate(critical_path):
            param_obj = registry.get_parameter(param)
            value_str = f" = {param_obj.value}" if param_obj and param_obj.value is not None else ""
            print(f"   {i+1}. {param}{value_str}")
        
        if mapper.critical_paths:
            cp = mapper.critical_paths[0]
            print(f"   Total influence: {cp.total_influence:.3f}")
            print(f"   Path type: {cp.path_type}")
    else:
        print("   No critical path found")
    
    # 6. Parameter clustering
    print("\n6. Parameter clustering analysis:")
    clusters = mapper.identify_parameter_clusters()
    
    print(f"   Found {len(clusters)} parameter clusters:")
    for cluster_id, params in clusters.items():
        print(f"   • {cluster_id}: {len(params)} parameters")
        print(f"     Parameters: {', '.join(params[:5])}")
        if len(params) > 5:
            print(f"     ... and {len(params) - 5} more")
    
    # Show detailed cluster information
    if mapper.clusters:
        print("\n   Detailed cluster analysis:")
        for cluster in mapper.clusters[:3]:  # Show first 3 clusters
            print(f"   Cluster {cluster.cluster_id}:")
            print(f"     Parameters: {len(cluster.parameters)}")
            print(f"     Coupling strength: {cluster.coupling_strength:.3f}")
            print(f"     Primary realm: {cluster.primary_realm}")
            print(f"     Description: {cluster.description}")
    
    # 7. Influence matrix analysis
    print("\n7. Parameter influence analysis:")
    influence_matrix = mapper.compute_influence_matrix()
    
    if influence_matrix.size > 0:
        print(f"   Influence matrix: {influence_matrix.shape[0]}×{influence_matrix.shape[1]}")
        print(f"   Total influence: {influence_matrix.sum():.3f}")
        print(f"   Max influence: {influence_matrix.max():.3f}")
        print(f"   Average influence: {influence_matrix.mean():.3f}")
        
        # Parameter influence ranking
        ranking = mapper.get_parameter_influence_ranking()
        print(f"\n   Top 5 most influential parameters:")
        for i, (param, influence) in enumerate(ranking[:5]):
            param_obj = registry.get_parameter(param)
            realm = param_obj.fixed_by_realm if param_obj else "unknown"
            print(f"   {i+1}. {param}: {influence:.3f} (realm: {realm})")
    
    # 8. Summary statistics
    print("\n8. Dependency graph statistics:")
    stats = mapper.get_summary_statistics()
    
    print(f"   • Total parameters: {stats['total_parameters']}")
    print(f"   • Total dependencies: {stats['total_dependencies']}")
    print(f"   • Total clusters: {stats['total_clusters']}")
    print(f"   • Graph density: {stats['graph_density']:.3f}")
    print(f"   • Is connected: {stats['is_connected']}")
    print(f"   • Has cycles: {stats['has_cycles']}")
    
    if 'most_influential_parameter' in stats:
        print(f"   • Most influential parameter: {stats['most_influential_parameter']}")
        print(f"   • Max influence: {stats['max_influence']:.3f}")
        print(f"   • Average influence: {stats['avg_influence']:.3f}")
    
    # 9. Realm-based analysis
    print("\n9. Realm-based dependency analysis:")
    if 'dependencies_by_realm' in stats:
        realm_deps = stats['dependencies_by_realm']
        sorted_realms = sorted(realm_deps.items(), key=lambda x: x[1], reverse=True)
        
        print("   Dependencies by realm:")
        for realm, count in sorted_realms[:8]:  # Show top 8 realms
            print(f"   • {realm}: {count} dependencies")
    
    # 10. Export analysis results
    print("\n10. Exporting dependency analysis results...")
    
    # Export graph as JSON
    try:
        mapper.export_graph('json', 'dependency_graph.json')
        print("   ✓ Exported dependency graph to 'dependency_graph.json'")
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
    
    # Create summary report
    summary_report = {
        'analysis_summary': {
            'total_parameters': len(registry.get_all_parameters()),
            'parameters_with_values': len([p for p in registry.get_all_parameters().values() if p.value is not None]),
            'total_dependencies': len(mapper.dependencies),
            'dependency_types': dep_types,
            'critical_path_length': len(critical_path),
            'cluster_count': len(clusters),
            'graph_statistics': stats
        },
        'critical_path': critical_path,
        'top_influential_parameters': ranking[:10] if 'ranking' in locals() else [],
        'clusters': {
            cluster_id: {
                'parameters': params,
                'size': len(params)
            }
            for cluster_id, params in clusters.items()
        }
    }
    
    with open('dependency_analysis_report.json', 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print("   ✓ Created summary report 'dependency_analysis_report.json'")
    
    print("\n=== Dependency Analysis Complete ===")
    print("\nKey insights:")
    print(f"• {len(mapper.dependencies)} parameter dependencies identified")
    print(f"• {len(clusters)} parameter clusters found")
    print(f"• Critical path contains {len(critical_path)} parameters")
    print(f"• Graph density: {stats['graph_density']:.1%} (how connected the parameters are)")
    
    if stats['has_cycles']:
        print("• Dependency graph contains cycles (complex interdependencies)")
    else:
        print("• Dependency graph is acyclic (clear execution order)")
    
    print("\nThe dependency analysis reveals the complex relationships between")
    print("coupling constants across QFD realms, enabling better understanding")
    print("of parameter space structure and optimization strategies.")


if __name__ == "__main__":
    main()