"""
Dependency mapper for coupling constants relationships.

This module provides tools for analyzing and visualizing the dependency
relationships between coupling constants across different QFD realms.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json

from ..registry.parameter_registry import ParameterRegistry, ConstraintType


@dataclass
class ParameterDependency:
    """Represents a dependency relationship between parameters."""
    source_param: str
    target_param: str
    dependency_type: str  # 'constraint', 'physics', 'realm_order', 'derived'
    strength: float  # 0.0 to 1.0, strength of dependency
    realm: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyCluster:
    """Represents a cluster of strongly coupled parameters."""
    cluster_id: str
    parameters: List[str]
    coupling_strength: float
    primary_realm: str
    description: str = ""


@dataclass
class CriticalPath:
    """Represents a critical path through the parameter dependency graph."""
    path_id: str
    parameters: List[str]
    total_influence: float
    path_type: str  # 'constraint', 'physics', 'execution'
    description: str = ""


class DependencyMapper:
    """
    Maps and analyzes dependencies between coupling constants.
    
    This class builds dependency graphs from parameter relationships,
    identifies critical paths, and finds parameter clusters.
    """
    
    def __init__(self, registry: ParameterRegistry):
        """
        Initialize dependency mapper.
        
        Args:
            registry: ParameterRegistry instance to analyze
        """
        self.registry = registry
        self.dependency_graph = nx.DiGraph()
        self.dependencies: List[ParameterDependency] = []
        self.clusters: List[DependencyCluster] = []
        self.critical_paths: List[CriticalPath] = []
        
    def build_dependency_graph(self) -> nx.DiGraph:
        """
        Build the complete dependency graph from parameter relationships.
        
        Returns:
            NetworkX directed graph of parameter dependencies
        """
        self.dependency_graph.clear()
        self.dependencies.clear()
        
        # Add all parameters as nodes
        all_params = self.registry.get_all_parameters()
        for param_name, param_state in all_params.items():
            self.dependency_graph.add_node(
                param_name,
                value=param_state.value,
                fixed_by=param_state.fixed_by_realm,
                constraint_count=len(param_state.constraints),
                change_count=len(param_state.history)
            )
        
        # Add constraint-based dependencies
        self._add_constraint_dependencies()
        
        # Add realm execution dependencies
        self._add_realm_dependencies()
        
        # Add physics-based dependencies (placeholder for future implementation)
        self._add_physics_dependencies()
        
        return self.dependency_graph
    
    def _add_constraint_dependencies(self) -> None:
        """Add dependencies based on parameter constraints."""
        all_params = self.registry.get_all_parameters()
        
        for param_name, param_state in all_params.items():
            for constraint in param_state.get_active_constraints():
                # Parameters with constraints from the same realm are related
                realm = constraint.realm
                
                # Find other parameters constrained by the same realm
                for other_param_name, other_param_state in all_params.items():
                    if other_param_name == param_name:
                        continue
                    
                    # Check if other parameter has constraints from same realm
                    other_realm_constraints = [
                        c for c in other_param_state.get_active_constraints()
                        if c.realm == realm
                    ]
                    
                    if other_realm_constraints:
                        # Create dependency based on constraint relationship
                        # Avoid self-loops
                        if param_name != other_param_name:
                            strength = self._calculate_constraint_strength(constraint, other_realm_constraints[0])
                            
                            dependency = ParameterDependency(
                                source_param=param_name,
                                target_param=other_param_name,
                                dependency_type='constraint',
                                strength=strength,
                                realm=realm,
                                description=f"Both constrained by {realm}"
                            )
                            
                            self.dependencies.append(dependency)
                            self.dependency_graph.add_edge(
                                param_name, other_param_name,
                                weight=strength,
                                type='constraint',
                                realm=realm
                            )
    
    def _add_realm_dependencies(self) -> None:
        """Add dependencies based on realm execution order."""
        all_params = self.registry.get_all_parameters()
        
        # Group parameters by the realm that last modified them
        params_by_realm = defaultdict(list)
        for param_name, param_state in all_params.items():
            if param_state.history:
                last_realm = param_state.history[-1].realm
                params_by_realm[last_realm].append(param_name)
            elif param_state.fixed_by_realm:
                # Also include parameters fixed by realm (even if no history)
                params_by_realm[param_state.fixed_by_realm].append(param_name)
        
        # Create dependencies based on typical realm execution order
        realm_order = [
            'config', 'cmb_config', 'ppn_config',
            'realm0_cmb', 'realm1_cosmic', 'realm2_dark_energy',
            'realm3_scales', 'realm4_em', 'realm5_electron',
            'realm6_leptons', 'realm7_proton', 'realm8_neutron',
            'realm9_deuteron', 'realm10_isotopes'
        ]
        
        # Add execution order dependencies
        for i, realm in enumerate(realm_order[:-1]):
            if realm in params_by_realm:
                for j, next_realm in enumerate(realm_order[i+1:], i+1):
                    if next_realm in params_by_realm:
                        # Parameters from earlier realms influence later ones
                        for source_param in params_by_realm[realm]:
                            for target_param in params_by_realm[next_realm]:
                                # Avoid self-loops
                                if source_param != target_param:
                                    strength = 0.3 / (j - i)  # Decay with distance
                                    
                                    dependency = ParameterDependency(
                                        source_param=source_param,
                                        target_param=target_param,
                                        dependency_type='realm_order',
                                        strength=strength,
                                        realm=f"{realm}->{next_realm}",
                                        description=f"Execution order: {realm} before {next_realm}"
                                    )
                                    
                                    self.dependencies.append(dependency)
                                    self.dependency_graph.add_edge(
                                        source_param, target_param,
                                        weight=strength,
                                        type='realm_order',
                                        realm_flow=f"{realm}->{next_realm}"
                                    )
                        break  # Only connect to immediate next realm with parameters
    
    def _add_physics_dependencies(self) -> None:
        """Add dependencies based on known physics relationships."""
        # This is a placeholder for physics-based dependencies
        # In a full implementation, this would include:
        # - PPN gamma/beta relationships
        # - CMB temperature/thermalization relationships
        # - Vacuum refractive index dependencies
        # - Coupling constant relationships from QFD theory
        
        # Example: PPN parameters are related
        ppn_gamma = self.registry.get_parameter("PPN_gamma")
        ppn_beta = self.registry.get_parameter("PPN_beta")
        
        if ppn_gamma and ppn_beta:
            dependency = ParameterDependency(
                source_param="PPN_gamma",
                target_param="PPN_beta",
                dependency_type='physics',
                strength=0.8,
                realm='ppn_theory',
                description="PPN parameters are physically related"
            )
            
            self.dependencies.append(dependency)
            self.dependency_graph.add_edge(
                "PPN_gamma", "PPN_beta",
                weight=0.8,
                type='physics',
                theory='ppn'
            )
        
        # Example: CMB temperature and thermalization
        t_cmb = self.registry.get_parameter("T_CMB_K")
        psi_s0 = self.registry.get_parameter("psi_s0")
        
        if t_cmb and psi_s0:
            dependency = ParameterDependency(
                source_param="psi_s0",
                target_param="T_CMB_K",
                dependency_type='physics',
                strength=0.9,
                realm='thermalization',
                description="Thermalization zeropoint determines CMB temperature"
            )
            
            self.dependencies.append(dependency)
            self.dependency_graph.add_edge(
                "psi_s0", "T_CMB_K",
                weight=0.9,
                type='physics',
                theory='thermalization'
            )
    
    def _calculate_constraint_strength(self, constraint1, constraint2) -> float:
        """Calculate the strength of dependency between two constraints."""
        # Base strength depends on constraint types
        type_strengths = {
            (ConstraintType.FIXED, ConstraintType.FIXED): 0.9,
            (ConstraintType.FIXED, ConstraintType.TARGET): 0.8,
            (ConstraintType.FIXED, ConstraintType.BOUNDED): 0.7,
            (ConstraintType.TARGET, ConstraintType.TARGET): 0.8,
            (ConstraintType.TARGET, ConstraintType.BOUNDED): 0.6,
            (ConstraintType.BOUNDED, ConstraintType.BOUNDED): 0.5,
        }
        
        key = (constraint1.constraint_type, constraint2.constraint_type)
        reverse_key = (constraint2.constraint_type, constraint1.constraint_type)
        
        return type_strengths.get(key, type_strengths.get(reverse_key, 0.3))
    
    def find_critical_path(self) -> List[str]:
        """
        Find the critical path through the dependency graph.
        
        Returns:
            List of parameter names in the critical path
        """
        if not self.dependency_graph.nodes():
            return []
        
        # Check if graph is acyclic first
        if nx.is_directed_acyclic_graph(self.dependency_graph):
            return self._find_critical_path_acyclic()
        else:
            return self._find_critical_path_with_cycles()
    
    def _find_critical_path_acyclic(self) -> List[str]:
        """Find critical path in acyclic graph."""
        try:
            # Use topological sort to find a valid ordering
            topo_order = list(nx.topological_sort(self.dependency_graph))
            
            # Find path with maximum total weight
            max_path = []
            max_weight = 0.0
            
            for start_node in topo_order[:5]:  # Check first few nodes as starting points
                for end_node in topo_order[-5:]:  # Check last few nodes as ending points
                    if start_node != end_node:
                        try:
                            path = nx.shortest_path(
                                self.dependency_graph, start_node, end_node, weight='weight'
                            )
                            
                            # Calculate total path weight
                            path_weight = 0.0
                            for i in range(len(path) - 1):
                                edge_data = self.dependency_graph.get_edge_data(path[i], path[i+1])
                                if edge_data:
                                    path_weight += edge_data.get('weight', 0.0)
                            
                            if path_weight > max_weight:
                                max_weight = path_weight
                                max_path = path
                                
                        except nx.NetworkXNoPath:
                            continue
            
            # Store as critical path
            if max_path:
                critical_path = CriticalPath(
                    path_id="main_critical_path",
                    parameters=max_path,
                    total_influence=max_weight,
                    path_type="execution",
                    description=f"Critical path with total influence {max_weight:.3f}"
                )
                self.critical_paths = [critical_path]
            
            return max_path
            
        except nx.NetworkXError:
            return self._find_critical_path_with_cycles()
    
    def _find_critical_path_with_cycles(self) -> List[str]:
        """Find critical path in graph with cycles."""
        # Use PageRank to find most influential nodes
        try:
            pagerank = nx.pagerank(self.dependency_graph, weight='weight')
            
            # Sort by influence
            sorted_params = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            
            # Take top influential parameters as critical path
            critical_path = [param for param, score in sorted_params[:5]]
            
            return critical_path
            
        except Exception:
            # Fallback: return parameters with most constraints
            all_params = self.registry.get_all_parameters()
            sorted_by_constraints = sorted(
                all_params.items(),
                key=lambda x: len(x[1].constraints),
                reverse=True
            )
            
            return [param_name for param_name, _ in sorted_by_constraints[:5]]
    
    def identify_parameter_clusters(self) -> Dict[str, List[str]]:
        """
        Identify clusters of strongly coupled parameters.
        
        Returns:
            Dictionary mapping cluster IDs to lists of parameter names
        """
        if not self.dependency_graph.nodes():
            return {}
        
        # Use community detection to find clusters
        try:
            # Convert to undirected graph for community detection
            undirected = self.dependency_graph.to_undirected()
            
            # Use greedy modularity communities
            communities = nx.community.greedy_modularity_communities(undirected, weight='weight')
            
            clusters = {}
            self.clusters = []
            
            for i, community in enumerate(communities):
                cluster_id = f"cluster_{i+1}"
                cluster_params = list(community)
                
                if len(cluster_params) > 1:  # Only include multi-parameter clusters
                    clusters[cluster_id] = cluster_params
                    
                    # Calculate average coupling strength within cluster
                    total_weight = 0.0
                    edge_count = 0
                    
                    for param1 in cluster_params:
                        for param2 in cluster_params:
                            if param1 != param2 and self.dependency_graph.has_edge(param1, param2):
                                edge_data = self.dependency_graph.get_edge_data(param1, param2)
                                total_weight += edge_data.get('weight', 0.0)
                                edge_count += 1
                    
                    avg_strength = total_weight / edge_count if edge_count > 0 else 0.0
                    
                    # Determine primary realm for cluster
                    realm_counts = defaultdict(int)
                    for param_name in cluster_params:
                        param = self.registry.get_parameter(param_name)
                        if param and param.fixed_by_realm:
                            realm_counts[param.fixed_by_realm] += 1
                    
                    primary_realm = max(realm_counts.items(), key=lambda x: x[1])[0] if realm_counts else "unknown"
                    
                    cluster_obj = DependencyCluster(
                        cluster_id=cluster_id,
                        parameters=cluster_params,
                        coupling_strength=avg_strength,
                        primary_realm=primary_realm,
                        description=f"Cluster of {len(cluster_params)} parameters with avg coupling {avg_strength:.3f}"
                    )
                    
                    self.clusters.append(cluster_obj)
            
            return clusters
            
        except Exception as e:
            # Fallback: group by realm
            return self._cluster_by_realm()
    
    def _cluster_by_realm(self) -> Dict[str, List[str]]:
        """Fallback clustering by realm."""
        clusters = defaultdict(list)
        
        all_params = self.registry.get_all_parameters()
        for param_name, param_state in all_params.items():
            if param_state.fixed_by_realm:
                clusters[f"realm_{param_state.fixed_by_realm}"].append(param_name)
            else:
                clusters["unconstrained"].append(param_name)
        
        # Filter out single-parameter clusters
        return {k: v for k, v in clusters.items() if len(v) > 1}
    
    def compute_influence_matrix(self) -> np.ndarray:
        """
        Compute the influence matrix between parameters.
        
        Returns:
            NxN matrix where element (i,j) represents influence of parameter i on parameter j
        """
        all_params = list(self.registry.get_all_parameters().keys())
        n_params = len(all_params)
        
        if n_params == 0:
            return np.array([])
        
        # Create parameter name to index mapping
        param_to_idx = {param: i for i, param in enumerate(all_params)}
        
        # Initialize influence matrix
        influence_matrix = np.zeros((n_params, n_params))
        
        # Fill matrix based on dependency graph
        for source, target, data in self.dependency_graph.edges(data=True):
            if source in param_to_idx and target in param_to_idx:
                source_idx = param_to_idx[source]
                target_idx = param_to_idx[target]
                weight = data.get('weight', 0.0)
                influence_matrix[source_idx, target_idx] = weight
        
        return influence_matrix
    
    def get_parameter_influence_ranking(self) -> List[Tuple[str, float]]:
        """
        Get parameters ranked by their total influence on other parameters.
        
        Returns:
            List of (parameter_name, total_influence) tuples, sorted by influence
        """
        influence_matrix = self.compute_influence_matrix()
        all_params = list(self.registry.get_all_parameters().keys())
        
        if influence_matrix.size == 0:
            return []
        
        # Calculate total outgoing influence for each parameter
        total_influences = np.sum(influence_matrix, axis=1)
        
        # Create ranking
        param_influences = [(all_params[i], total_influences[i]) for i in range(len(all_params))]
        param_influences.sort(key=lambda x: x[1], reverse=True)
        
        return param_influences
    
    def export_graph(self, format: str, filename: str) -> None:
        """
        Export the dependency graph in various formats.
        
        Args:
            format: Export format ('json', 'gml', 'graphml')
            filename: Output filename
        """
        if format.lower() == 'json':
            # Export as JSON with custom serialization
            graph_data = {
                'nodes': [
                    {
                        'id': node,
                        'attributes': dict(self.dependency_graph.nodes[node])
                    }
                    for node in self.dependency_graph.nodes()
                ],
                'edges': [
                    {
                        'source': source,
                        'target': target,
                        'attributes': dict(data)
                    }
                    for source, target, data in self.dependency_graph.edges(data=True)
                ],
                'dependencies': [
                    {
                        'source_param': dep.source_param,
                        'target_param': dep.target_param,
                        'dependency_type': dep.dependency_type,
                        'strength': dep.strength,
                        'realm': dep.realm,
                        'description': dep.description,
                        'metadata': dep.metadata
                    }
                    for dep in self.dependencies
                ],
                'clusters': [
                    {
                        'cluster_id': cluster.cluster_id,
                        'parameters': cluster.parameters,
                        'coupling_strength': cluster.coupling_strength,
                        'primary_realm': cluster.primary_realm,
                        'description': cluster.description
                    }
                    for cluster in self.clusters
                ]
            }
            
            with open(filename, 'w') as f:
                json.dump(graph_data, f, indent=2)
                
        elif format.lower() == 'gml':
            nx.write_gml(self.dependency_graph, filename)
            
        elif format.lower() == 'graphml':
            nx.write_graphml(self.dependency_graph, filename)
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about the dependency graph."""
        if not self.dependency_graph.nodes():
            return {'error': 'No dependency graph built'}
        
        stats = {
            'total_parameters': len(self.dependency_graph.nodes()),
            'total_dependencies': len(self.dependency_graph.edges()),
            'total_clusters': len(self.clusters),
            'graph_density': nx.density(self.dependency_graph),
            'is_connected': nx.is_weakly_connected(self.dependency_graph),
            'has_cycles': not nx.is_directed_acyclic_graph(self.dependency_graph),
        }
        
        # Dependency type breakdown
        dep_types = defaultdict(int)
        for dep in self.dependencies:
            dep_types[dep.dependency_type] += 1
        stats['dependency_types'] = dict(dep_types)
        
        # Realm breakdown
        realm_counts = defaultdict(int)
        for dep in self.dependencies:
            realm_counts[dep.realm] += 1
        stats['dependencies_by_realm'] = dict(realm_counts)
        
        # Influence statistics
        influences = self.get_parameter_influence_ranking()
        if influences:
            stats['most_influential_parameter'] = influences[0][0]
            stats['max_influence'] = influences[0][1]
            stats['avg_influence'] = np.mean([inf for _, inf in influences])
        
        return stats