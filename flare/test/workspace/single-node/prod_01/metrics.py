#!/usr/bin/env python3
"""
Extract federated learning metrics from NVIDIA FLARE logs and compute weighted averages (FedAvg style)
"""

import re
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import argparse


def extract_metrics_from_log(log_file: str) -> List[Dict]:
    """Extract training metrics from a single client log file."""
    metrics = []
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Pattern to match the summary lines with all metrics
        pattern = r'Sent model update to server\. Train loss: ([\d.]+), Test accuracy: ([\d.]+)%'
        matches = re.findall(pattern, content)
        
        # Also extract test loss separately
        test_loss_pattern = r'Test Loss: ([\d.]+), Accuracy: ([\d.]+)%'
        test_matches = re.findall(test_loss_pattern, content)
        
        # Extract number of samples (should be consistent per client)
        samples_pattern = r'Training samples: (\d+)'
        samples_match = re.search(samples_pattern, content)
        num_samples = int(samples_match.group(1)) if samples_match else 1
        
        for i, (train_loss, accuracy) in enumerate(matches):
            test_loss = test_matches[i][0] if i < len(test_matches) else 0.0
            
            metrics.append({
                'round': i + 1,
                'train_loss': float(train_loss),
                'test_loss': float(test_loss),
                'accuracy': float(accuracy),
                'num_samples': num_samples
            })
    
    except FileNotFoundError:
        print(f"Warning: Log file not found: {log_file}")
    except Exception as e:
        print(f"Error processing {log_file}: {e}")
    
    return metrics

def extract_all_client_metrics(base_path: str = ".",run: str = "") -> Dict[str, List[Dict]]:
    """Extract metrics from all client log files."""
    client_metrics = {}
    
    # Look for client directories
    base_path = Path(base_path)
    for client_dir in base_path.glob("site-*"):
        if client_dir.is_dir():
            site_name = client_dir.name
            log_file = client_dir / run / "log_fl.txt"
            
            print(f"Processing {site_name} from {log_file}")
            metrics = extract_metrics_from_log(str(log_file))
            
            if metrics:
                client_metrics[site_name] = metrics
                print(f"  Found {len(metrics)} rounds of data")
            else:
                print(f"  No metrics found")
    
    return client_metrics

def compute_fedavg_metrics(client_metrics: Dict[str, List[Dict]]) -> List[Dict]:
    """Compute FedAvg weighted averages across all clients."""
    if not client_metrics:
        return []
    
    # Determine the number of rounds
    max_rounds = max(len(metrics) for metrics in client_metrics.values())
    fedavg_metrics = []
    
    for round_num in range(1, max_rounds + 1):
        round_data = []
        total_samples = 0
        
        # Collect data from all clients for this round
        for site_name, metrics in client_metrics.items():
            if round_num <= len(metrics):
                round_data.append(metrics[round_num - 1])
                total_samples += metrics[round_num - 1]['num_samples']
        
        if not round_data:
            continue
        
        # Compute weighted averages
        weighted_train_loss = sum(
            data['train_loss'] * data['num_samples'] for data in round_data
        ) / total_samples
        
        weighted_test_loss = sum(
            data['test_loss'] * data['num_samples'] for data in round_data
        ) / total_samples
        
        weighted_accuracy = sum(
            data['accuracy'] * data['num_samples'] for data in round_data
        ) / total_samples
        
        fedavg_metrics.append({
            'round': round_num,
            'train_loss': weighted_train_loss,
            'test_loss': weighted_test_loss,
            'accuracy': weighted_accuracy,
            'total_samples': total_samples,
            'num_clients': len(round_data)
        })
    
    return fedavg_metrics

def save_metrics_summary(client_metrics: Dict[str, List[Dict]], 
                        fedavg_metrics: List[Dict], 
                        output_file: str = "federated_metrics.json"):
    """Save all metrics to a JSON file."""
    summary = {
        'client_metrics': client_metrics,
        'fedavg_metrics': fedavg_metrics,
        'metadata': {
            'num_clients': len(client_metrics),
            'num_rounds': len(fedavg_metrics),
            'clients': list(client_metrics.keys())
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Metrics saved to {output_file}")

def plot_fedavg_metrics(client_metrics: Dict[str, List[Dict]], 
                       fedavg_metrics: List[Dict], 
                       save_plots: bool = True):
    """Create comprehensive plots of the federated learning metrics."""
    
    if not fedavg_metrics:
        print("No metrics to plot")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Federated Learning Metrics (Fashion-MNIST CNN)', fontsize=16, fontweight='bold')
    
    rounds = [m['round'] for m in fedavg_metrics]
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    train_losses = [m['train_loss'] for m in fedavg_metrics]
    ax1.plot(rounds, train_losses, 'b-o', linewidth=2, markersize=6, label='FedAvg (Weighted)')
    
    # Plot individual client training losses
    for site_name, metrics in client_metrics.items():
        site_rounds = [m['round'] for m in metrics]
        site_train_losses = [m['train_loss'] for m in metrics]
        ax1.plot(site_rounds, site_train_losses, '--', alpha=0.6, label=site_name)
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Over Rounds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test Loss
    ax2 = axes[0, 1]
    test_losses = [m['test_loss'] for m in fedavg_metrics]
    ax2.plot(rounds, test_losses, 'r-o', linewidth=2, markersize=6, label='FedAvg (Weighted)')
    
    # Plot individual client test losses
    for site_name, metrics in client_metrics.items():
        site_rounds = [m['round'] for m in metrics]
        site_test_losses = [m['test_loss'] for m in metrics]
        ax2.plot(site_rounds, site_test_losses, '--', alpha=0.6, label=site_name)
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss Over Rounds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy
    ax3 = axes[1, 0]
    accuracies = [m['accuracy'] for m in fedavg_metrics]
    ax3.plot(rounds, accuracies, 'g-o', linewidth=2, markersize=6, label='FedAvg (Weighted)')
    
    # Plot individual client accuracies
    for site_name, metrics in client_metrics.items():
        site_rounds = [m['round'] for m in metrics]
        site_accuracies = [m['accuracy'] for m in metrics]
        ax3.plot(site_rounds, site_accuracies, '--', alpha=0.6, label=site_name)
    
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Test Accuracy Over Rounds')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary table as text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    summary_text = "FedAvg Summary:\n\n"
    for i, m in enumerate(fedavg_metrics):
        summary_text += f"Round {m['round']}:\n"
        summary_text += f"  Train Loss: {m['train_loss']:.4f}\n"
        summary_text += f"  Test Loss: {m['test_loss']:.4f}\n"
        summary_text += f"  Accuracy: {m['accuracy']:.2f}%\n"
        summary_text += f"  Clients: {m['num_clients']}\n\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('federated_learning_metrics.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'federated_learning_metrics.png'")
    
    plt.show()

def print_metrics_table(fedavg_metrics: List[Dict]):
    """Print a formatted table of FedAvg metrics."""
    if not fedavg_metrics:
        print("No metrics to display")
        return
    
    print("\n" + "="*80)
    print("FEDERATED LEARNING METRICS SUMMARY (FedAvg Weighted Averages)")
    print("="*80)
    print(f"{'Round':<6} {'Train Loss':<12} {'Test Loss':<12} {'Accuracy':<12} {'Clients':<8}")
    print("-"*80)
    
    for m in fedavg_metrics:
        print(f"{m['round']:<6} {m['train_loss']:<12.4f} {m['test_loss']:<12.4f} "
              f"{m['accuracy']:<12.2f} {m['num_clients']:<8}")
    
    print("="*80)
    
    # Print improvement metrics
    if len(fedavg_metrics) > 1:
        first_round = fedavg_metrics[0]
        last_round = fedavg_metrics[-1]
        
        print(f"\nIMPROVEMENT FROM ROUND 1 TO {last_round['round']}:")
        print(f"  Train Loss: {first_round['train_loss']:.4f} → {last_round['train_loss']:.4f} "
              f"({last_round['train_loss'] - first_round['train_loss']:+.4f})")
        print(f"  Test Loss:  {first_round['test_loss']:.4f} → {last_round['test_loss']:.4f} "
              f"({last_round['test_loss'] - first_round['test_loss']:+.4f})")
        print(f"  Accuracy:   {first_round['accuracy']:.2f}% → {last_round['accuracy']:.2f}% "
              f"({last_round['accuracy'] - first_round['accuracy']:+.2f}%)")

def main():
    """Main function to extract and analyze federated learning metrics."""
    print("Extracting NVIDIA FLARE Federated Learning Metrics...")
    print("-" * 60)
    parser = argparse.ArgumentParser(description="Extract and analyze FLARE federated learning metrics")
    parser.add_argument("--base_path", type=str, default=".", help="Base path to search for client directories")
    parser.add_argument("--run", type=str, default="", help="Run identifier to filter logs (e.g., 'run1')")
    args = parser.parse_args()
    
    # Extract metrics from all client logs
    client_metrics = extract_all_client_metrics(base_path=args.base_path, run=args.run)
    
    if not client_metrics:
        print("No client metrics found. Please check that client log files exist.")
        return
    
    # Compute FedAvg weighted averages
    fedavg_metrics = compute_fedavg_metrics(client_metrics)
    
    # Print summary table
    print_metrics_table(fedavg_metrics)
    
    # Save metrics to JSON
    save_metrics_summary(client_metrics, fedavg_metrics)
    
    # Create plots
    try:
        plot_fedavg_metrics(client_metrics, fedavg_metrics)
    except ImportError:
        print("\nMatplotlib not available. Install with: pip install matplotlib")
        print("Metrics have been saved to federated_metrics.json")

if __name__ == "__main__":
    main()