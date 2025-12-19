"""Snapshot generator for visualization"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os


class StepSnapshot:
    def __init__(self, n_edges: int, service_names: List[str], snapshot_dir: str = "episode_snapshots"):
        self.n_edges = n_edges
        self.service_names = service_names
        self.n_services = len(service_names)
        self.snapshot_dir = snapshot_dir
        
        # Create directory
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Colors
        self.service_colors = plt.cm.Set3(np.linspace(0, 1, self.n_services))
        self.static_color = '#6c757d'  # Gray for static energy
    
    def create_step_snapshot(self, step: int, analysis: Dict) -> str:
        """Create and save snapshot image for current step"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'System State - Step {step}', fontsize=16, fontweight='bold')
        
        # 1. Heatmap of services per edge
        self._create_services_heatmap(axes[0, 0], analysis)
        
        # 2. Energy consumed
        self._create_energy_plot(axes[0, 1], analysis)
        
        # 3. Cloud status
        self._create_cloud_status(axes[1, 1], analysis)
        
        plt.tight_layout()
        
        # Save image
        filename = f"{self.snapshot_dir}/step_{step:04d}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_services_heatmap(self, ax, analysis: Dict):
        """Create heatmap showing service state on each edge"""
        heatmap_data = np.zeros((self.n_edges, self.n_services))
        edge_labels = []
        
        # Check actual number of edges
        actual_edges = len(analysis['edges'])
        if actual_edges != self.n_edges:
            print(f"⚠️ Warning: Expected {self.n_edges} edges, found {actual_edges} in analysis")
        
        # Fill heatmap data
        for edge_id, edge_data in analysis['edges'].items():
            if edge_id < self.n_edges:
                edge_labels.append(f"Edge {edge_id}")
                for service_idx, service_name in enumerate(self.service_names):
                    # Check service state
                    is_used = any(s['name'] == service_name for s in edge_data['used_services'])
                    is_unused = any(s['name'] == service_name for s in edge_data['unused_services'])
                    
                    if is_used:
                        heatmap_data[edge_id, service_idx] = 2  # Used
                    elif is_unused:
                        heatmap_data[edge_id, service_idx] = 1  # Not used
                    else:
                        heatmap_data[edge_id, service_idx] = 0  # Not deployed
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)
        
        # Configure axes
        ax.set_xticks(range(self.n_services))
        ax.set_xticklabels(self.service_names, rotation=45, ha='right')
        ax.set_yticks(range(self.n_edges))
        ax.set_yticklabels(edge_labels)
        
        # Add annotations
        for i in range(self.n_edges):
            for j in range(self.n_services):
                text = ax.text(j, i, ['-', 'NO', 'YES'][int(heatmap_data[i, j])],
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Service State by Edge\n(Used/Not Used/Not Deployed)', 
                    fontweight='bold')
        ax.set_xlabel('Services')
        ax.set_ylabel('Edges')
        
        # Color bar
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(['Not deployed', 'Not used', 'Used'])
    
    def _create_energy_plot(self, ax, analysis: Dict):
        """Create plot showing energy decomposed into static/useful/idle"""
        edges_energy = []
        edge_labels = []
        static_energies = []
        dynamic_energies = []
        service_counts = []
        
        # Collect edge data
        for edge_id, edge_data in analysis['edges'].items():
            if edge_id != analysis['cloud']['cloud_id']:  # Exclude cloud
                edges_energy.append(edge_data['energy_consumed'])
                edge_labels.append(f"Edge {edge_id}")
                service_counts.append(edge_data['total_services'])
                
                # Get static and dynamic energy
                breakdown = edge_data['energy_breakdown']
                static_energies.append(breakdown['static'])
                dynamic_energies.append(breakdown['dynamic'])
        
        # Add cloud energy
        edges_energy.append(analysis['cloud']['energy_consumed'])
        service_counts.append(self.n_services)
        edge_labels.append("Cloud")
        
        # For cloud
        static_energies.append(analysis['cloud']['base_energy'])
        dynamic_energies.append(analysis['cloud']['energy_consumed'] - analysis['cloud']['base_energy'])
        
        n_edges = len(edges_energy) - 1  # Number of edges (without cloud)
        x_positions = range(len(edge_labels))
        
        # Create simplified stacked plot
        self._create_simplified_stacked_plot(ax, x_positions, edge_labels, 
                                           static_energies, dynamic_energies, 
                                           edges_energy, n_edges, analysis)
        
        # Configure main axis (energy)
        ax.set_ylabel('Energy (Joules)', fontweight='bold')
        ax.set_xlabel('Compute Nodes', fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(edge_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Create second axis for deployed services
        ax2 = ax.twinx()
        line = ax2.plot(x_positions, service_counts, 'o-', color='#2ecc71', linewidth=3, 
                       markersize=8, label='Deployed services', alpha=0.8)
        
        # Add values on service line
        for i, count in enumerate(service_counts):
            ax2.text(i, count + 0.1, f'{count}', ha='center', va='bottom', 
                    fontweight='bold', color='#2ecc71', fontsize=9)
        
        # Configure second axis (services)
        ax2.set_ylabel('Number of Deployed Services', fontweight='bold', color='#2ecc71')
        ax2.tick_params(axis='y', labelcolor='#2ecc71')
        
        # Adjust axis limits
        if edges_energy:
            ax.set_ylim(0, max(edges_energy) * 1.15)
        else:
            ax.set_ylim(0, 1)
            
        if service_counts:
            ax2.set_ylim(0, max(service_counts) * 1.2)
        else:
            ax2.set_ylim(0, 5)
        
        # Combined legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', framealpha=0.9)
        
        # Title
        ax.set_title('Energy Consumed (Decomposed) and Deployed Services per Node', 
                    fontweight='bold', fontsize=12, pad=20)
    
    def _create_simplified_stacked_plot(self, ax, x_positions, edge_labels, 
                                      static_energies, dynamic_energies, 
                                      total_energy, n_edges, analysis_data):
        """Create stacked bar chart"""
        # Convert to numpy arrays
        static_energies = np.array(static_energies, dtype=np.float64)
        dynamic_energies = np.array(dynamic_energies, dtype=np.float64)
        
        # Extract components (useful vs idle, but include startup in idle for display)
        dynamic_useful = []
        dynamic_idle_combined = []  # Combined idle + startup
        
        for i, edge_label in enumerate(edge_labels):
            if i < n_edges:  # It's an edge
                edge_id = int(edge_label.split(' ')[1])  # Extract edge ID
                if edge_id in analysis_data['edges']:
                    edge_analysis = analysis_data['edges'][edge_id]
                    breakdown = edge_analysis['energy_breakdown']
                    dynamic_useful.append(breakdown['dynamic_useful'])
                    # Combine idle and startup for display
                    combined_idle = breakdown['dynamic_idle'] + breakdown.get('startup_penalty', 0)
                    dynamic_idle_combined.append(combined_idle)
                else:
                    dynamic_useful.append(dynamic_energies[i])
                    dynamic_idle_combined.append(0)
            else:  # It's the cloud
                cloud_analysis = analysis_data['cloud']
                dynamic_useful.append(cloud_analysis['useful_energy'])
                dynamic_idle_combined.append(cloud_analysis['waste_penalty'])
        
        dynamic_useful = np.array(dynamic_useful)
        dynamic_idle_combined = np.array(dynamic_idle_combined)
        
        # Colors
        static_color = '#6c757d'    # Gray for static
        useful_color = '#2ecc71'    # Green for useful energy
        idle_color = '#e74c3c'      # Red for idle energy
        
        # 1. Base bar: static energy
        bottom_bars = ax.bar(x_positions, static_energies, 
                           color=static_color, alpha=0.8,
                           label='Static Energy', edgecolor='white', linewidth=0.5)
        
        # 2. Bars for useful dynamic energy
        useful_bars = ax.bar(x_positions, dynamic_useful, bottom=static_energies,
                            color=useful_color, alpha=0.8,
                            label='Useful Energy', edgecolor='white', linewidth=0.5)
        
        # 3. Bars for combined idle energy
        idle_bars = ax.bar(x_positions, dynamic_idle_combined, 
                          bottom=static_energies + dynamic_useful,
                          color=idle_color, alpha=0.8,
                          label='Idle Energy', edgecolor='white', linewidth=0.5)
        
        # 4. Add total value labels
        self._add_total_energy_annotations(ax, x_positions, total_energy)
        
        # 5. Add composition percentages
        self._add_composition_annotations_simple(ax, x_positions, static_energies, 
                                               dynamic_useful, dynamic_idle_combined, total_energy)
    
    def _add_total_energy_annotations(self, ax, x_positions, total_energy):
        """Add total energy values on bars"""
        if not total_energy:
            return
            
        max_energy = max(total_energy) if total_energy else 1
        
        for i, energy in enumerate(total_energy):
            if i < len(x_positions):
                ax.text(x_positions[i], energy + max_energy * 0.01,
                       f'{energy:.0f} J', ha='center', va='bottom', 
                       fontweight='bold', fontsize=9, color='black')
    
    def _add_composition_annotations_simple(self, ax, x_positions, static_energies, 
                                          dynamic_useful, dynamic_idle_combined, total_energy):
        """Add composition percentages showing only 3 components"""
        for i in range(len(x_positions)):
            if i < len(total_energy) and total_energy[i] > 0:
                static_pct = (static_energies[i] / total_energy[i]) * 100
                useful_pct = (dynamic_useful[i] / total_energy[i]) * 100
                idle_pct = (dynamic_idle_combined[i] / total_energy[i]) * 100
                
                # Annotation for static energy
                if static_energies[i] > 0:
                    static_height = static_energies[i] / 2
                    ax.text(x_positions[i], static_height, f'{static_pct:.0f}%', 
                           ha='center', va='center', fontsize=8, color='white',
                           fontweight='bold')
                
                # Annotation for useful energy
                if dynamic_useful[i] > 0:
                    useful_height = static_energies[i] + (dynamic_useful[i] / 2)
                    ax.text(x_positions[i], useful_height, f'{useful_pct:.0f}%', 
                           ha='center', va='center', fontsize=8, color='white',
                           fontweight='bold')
                
                # Annotation for combined idle energy
                if dynamic_idle_combined[i] > 0:
                    idle_height = static_energies[i] + dynamic_useful[i] + (dynamic_idle_combined[i] / 2)
                    ax.text(x_positions[i], idle_height, f'{idle_pct:.0f}%', 
                           ha='center', va='center', fontsize=8, color='white',
                           fontweight='bold')
    
    def _create_cloud_status(self, ax, analysis: Dict):
        """Create status panel for cloud"""
        cloud_data = analysis['cloud']
        
        # Create textual table
        cloud_info = [
            f"Vehicles served: {cloud_data['vehicles_served']}",
            f"Energy consumed: {cloud_data['energy_consumed']:.0f} J",
            f"Useful CPU: {cloud_data['useful_cpu']:.1f} units",
            f"Wasted CPU: {cloud_data['wasted_cpu']:.1f} units",
            f"Services used: {', '.join(cloud_data['services_used']) if cloud_data['services_used'] else 'None'}",
            f"Available services: {len(cloud_data['services_unused'])}"
        ]
        
        # Create table
        table_data = []
        for info in cloud_info:
            table_data.append([info])
        
        # Create table
        table = ax.table(cellText=table_data, 
                        cellLoc='left',
                        loc='center',
                        bbox=[0.1, 0.1, 0.8, 0.8])
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style cells
        for i in range(len(cloud_info)):
            table[(i, 0)].set_facecolor('#f8f9fa')
            table[(i, 0)].set_edgecolor('#dee2e6')
        
        ax.set_title('Cloud Status', fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add energy summary
        energy_text = (
            f"Energy detail:\n"
            f"• Useful: {cloud_data['useful_energy']:.0f} J\n"
            f"• Waste: {cloud_data['waste_penalty']:.0f} J\n"
            f"• Fixed: {cloud_data['base_energy']} J"
        )
        ax.text(0.5, 0.02, energy_text, transform=ax.transAxes, 
               ha='center', va='bottom', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#e9ecef", alpha=0.7))
    
    def create_combined_analysis(self, all_analysis: List[Dict]):
        """Create combined analysis of entire episode"""
        if not all_analysis:
            return
        
        steps = [analysis['step'] for analysis in all_analysis]
        
        # Create figure with 1 row, 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Complete Analysis of Last Episode', fontsize=16, fontweight='bold')
        
        # 1. Total energy evolution
        total_energy = [analysis['summary']['total_energy'] for analysis in all_analysis]
        ax1.plot(steps, total_energy, 'o-', linewidth=2, color='#e74c3c')
        ax1.set_title('Total Energy Evolution', fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Energy (J)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Energy distribution over ENTIRE episode (pie chart)
        self._create_episode_energy_pie_chart(ax2, all_analysis)
        
        plt.tight_layout()
        plt.savefig(f"{self.snapshot_dir}/episode_analysis_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_episode_energy_pie_chart(self, ax, all_analysis: List[Dict]):
        """Create pie chart of energy distribution over entire episode"""
        # Calculate cumulative energy
        edge_energy_totals = {}
        cloud_energy_total = 0
        
        # Initialize counters
        for analysis in all_analysis:
            # Cumulative cloud energy
            cloud_energy_total += analysis['cloud']['energy_consumed']
            
            # Cumulative edge energy
            for edge_id, edge_data in analysis['edges'].items():
                if edge_id not in edge_energy_totals:
                    edge_energy_totals[edge_id] = 0
                edge_energy_totals[edge_id] += edge_data['energy_consumed']
        
        # Prepare data for pie chart
        energy_labels = []
        energy_values = []
        colors = []
        
        # Add each edge
        edge_colors = plt.cm.Set3(np.linspace(0, 1, len(edge_energy_totals)))
        for i, (edge_id, total_energy) in enumerate(edge_energy_totals.items()):
            energy_labels.append(f'Edge {edge_id}')
            energy_values.append(total_energy)
            colors.append(edge_colors[i])
        
        # Add cloud
        energy_labels.append('Cloud')
        energy_values.append(cloud_energy_total)
        colors.append('#9b59b6')  # Purple for cloud
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            energy_values, 
            labels=energy_labels, 
            colors=colors, 
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 9}
        )
        
        # Improve readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Detailed Energy Distribution', fontweight='bold')