#!/usr/bin/env python3.11
"""
Training Monitor - Real-time monitoring of model training progress
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not installed. Install with: pip install rich")


class TrainingMonitor:
    """Monitor training progress and display metrics."""
    
    def __init__(self, log_file: Path = None):
        self.log_file = log_file or Path("training_optimized.log")
        self.console = Console() if RICH_AVAILABLE else None
        self.metrics_history: List[Dict[str, Any]] = []
        
    def parse_log_line(self, line: str) -> Dict[str, Any]:
        """Extract metrics from log line."""
        metrics = {}
        
        # Extract accuracy
        if "Accuracy:" in line:
            try:
                metrics['accuracy'] = float(line.split("Accuracy:")[1].split()[0])
            except (IndexError, ValueError):
                pass
        
        # Extract ROC-AUC
        if "ROC-AUC:" in line:
            try:
                metrics['roc_auc'] = float(line.split("ROC-AUC:")[1].split()[0])
            except (IndexError, ValueError):
                pass
        
        # Extract sensitivity
        if "Sensitivity:" in line:
            try:
                metrics['sensitivity'] = float(line.split("Sensitivity:")[1].split()[0])
            except (IndexError, ValueError):
                pass
        
        # Extract specificity
        if "Specificity:" in line:
            try:
                metrics['specificity'] = float(line.split("Specificity:")[1].split()[0])
            except (IndexError, ValueError):
                pass
        
        # Extract ECE
        if "ECE:" in line:
            try:
                metrics['ece'] = float(line.split("ECE:")[1].split()[0])
            except (IndexError, ValueError):
                pass
        
        return metrics
    
    def create_metrics_table(self, latest_metrics: Dict[str, Any]) -> Table:
        """Create a formatted table of metrics."""
        table = Table(title="Training Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=15)
        table.add_column("Status", style="yellow", width=15)
        
        metrics_config = [
            ("Accuracy", "accuracy", 0.92),
            ("ROC-AUC", "roc_auc", 0.90),
            ("Sensitivity", "sensitivity", 0.85),
            ("Specificity", "specificity", 0.80),
            ("ECE", "ece", 0.10, True),  # Lower is better
        ]
        
        for name, key, threshold, *lower_better in metrics_config:
            if key in latest_metrics:
                value = latest_metrics[key]
                value_str = f"{value:.4f}"
                
                # Determine status
                is_lower_better = len(lower_better) > 0 and lower_better[0]
                if is_lower_better:
                    status = "✓ Good" if value <= threshold else "⚠ High"
                else:
                    status = "✓ Good" if value >= threshold else "⚠ Low"
                
                table.add_row(name, value_str, status)
            else:
                table.add_row(name, "N/A", "Pending")
        
        return table
    
    def monitor_file(self, update_interval: float = 2.0):
        """Monitor log file and display real-time updates."""
        if not self.log_file.exists():
            print(f"Log file not found: {self.log_file}")
            print("Waiting for training to start...")
        
        latest_metrics = {}
        last_position = 0
        
        if RICH_AVAILABLE:
            with Live(self.create_metrics_table(latest_metrics), refresh_per_second=1) as live:
                while True:
                    if self.log_file.exists():
                        with open(self.log_file, 'r') as f:
                            f.seek(last_position)
                            new_lines = f.readlines()
                            last_position = f.tell()
                            
                            for line in new_lines:
                                metrics = self.parse_log_line(line)
                                if metrics:
                                    latest_metrics.update(metrics)
                                    self.metrics_history.append({
                                        'timestamp': time.time(),
                                        **metrics
                                    })
                            
                            if new_lines:
                                live.update(self.create_metrics_table(latest_metrics))
                    
                    time.sleep(update_interval)
        else:
            # Fallback to simple text output
            while True:
                if self.log_file.exists():
                    with open(self.log_file, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()
                        
                        for line in new_lines:
                            metrics = self.parse_log_line(line)
                            if metrics:
                                latest_metrics.update(metrics)
                                print(f"\n[{time.strftime('%H:%M:%S')}] Metrics Update:")
                                for key, value in metrics.items():
                                    print(f"  {key}: {value:.4f}")
                
                time.sleep(update_interval)
    
    def generate_summary(self, output_file: Path = None):
        """Generate training summary report."""
        if not self.metrics_history:
            print("No metrics collected yet.")
            return
        
        summary = {
            "total_updates": len(self.metrics_history),
            "final_metrics": self.metrics_history[-1] if self.metrics_history else {},
            "best_metrics": {},
            "history": self.metrics_history
        }
        
        # Find best values
        for key in ['accuracy', 'roc_auc', 'sensitivity', 'specificity']:
            values = [m.get(key) for m in self.metrics_history if key in m]
            if values:
                summary['best_metrics'][key] = max(values)
        
        # ECE - lower is better
        ece_values = [m.get('ece') for m in self.metrics_history if 'ece' in m]
        if ece_values:
            summary['best_metrics']['ece'] = min(ece_values)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Summary saved to: {output_file}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--log-file", type=Path, default="training_optimized.log",
                        help="Path to training log file")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Update interval in seconds")
    parser.add_argument("--summary", type=Path, help="Generate summary and save to file")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(log_file=args.log_file)
    
    if args.summary:
        monitor.generate_summary(output_file=args.summary)
    else:
        try:
            monitor.monitor_file(update_interval=args.interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            if monitor.metrics_history:
                print("\nGenerating summary...")
                summary = monitor.generate_summary()
                print(json.dumps(summary['final_metrics'], indent=2))


if __name__ == "__main__":
    main()
