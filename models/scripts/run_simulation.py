#!/usr/bin/env python3
"""Main script to run the simulation"""
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from vehicular_network.simulator import CombinedDocumentSimulator
from vehicular_network.visualization import visualize_all_nodes_features


def main():
    parser = argparse.ArgumentParser(description="Run vehicular network simulation")
    parser.add_argument('--network', type=str, default='data/network/manhattan.net.xml',
                       help="SUMO network file")
    parser.add_argument('--num-edges', type=int, default=7,
                       help="Number of edge nodes")
    parser.add_argument('--vehicles', type=int, default=20,
                       help="Target number of vehicles")
    parser.add_argument('--steps', type=int, default=200,
                       help="Number of time steps")
    parser.add_argument('--gui', action='store_true',
                       help="Use SUMO GUI")
    parser.add_argument('--output', type=str, default='data/graphs/manhattan_snapshots.bin',
                       help="Output file for snapshots")
    parser.add_argument('--visualize', action='store_true',
                       help="Visualize snapshots")
    
    args = parser.parse_args()
    
    print("🚀 Starting vehicular network simulation")
    
    # Create simulator
    simulator = CombinedDocumentSimulator(
        num_edges=args.num_edges,
        target_vehicles=args.vehicles,
        time_steps=args.steps,
        network_file=args.network,
        use_gui=args.gui,
        save_snapshots=True,
        output_file=args.output
    )
    
    # Generate snapshots
    print("⏳ Generating snapshots...")
    
    graphs = []
    for t in range(simulator.time_steps):
        g = simulator.generate_snapshot(t)
        if t == 0 and g.number_of_nodes('vehicle') == 0:
            continue
        graphs.append(g)
    
    # Save snapshots
    simulator.saved_snapshots = graphs
    simulator.save_snapshots_to_file()
    
    # Visualize if requested
    if args.visualize:
        print("\n📊 Visualizing features...")
        visualize_all_nodes_features(graphs)
        
        # Visualize selected snapshots
        for t in [0, 50, 100, 150]:
            if t < len(graphs):
                print(f"\n🎨 Visualizing snapshot {t}")
                simulator.visualize_snapshot(graphs[t], t)
    
    print(f"✅ {len(graphs)} snapshots saved to {args.output}")


if __name__ == "__main__":
    main()