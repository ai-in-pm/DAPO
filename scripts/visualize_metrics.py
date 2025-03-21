import os
import argparse
import sqlite3
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import DAPODatabase
from utils.logging import setup_logger

def load_metrics_from_database(db_path: str, metric_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load metrics from the DAPO database.
    
    Args:
        db_path: Path to the database file.
        metric_name: Optional filter for specific metric name.
        
    Returns:
        List of metric dictionaries.
    """
    database = DAPODatabase(db_path)
    return database.get_metrics(metric_name=metric_name, limit=1000)

def load_metrics_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load metrics from a JSON or JSONL file.
    
    Args:
        file_path: Path to the metrics file.
        
    Returns:
        List of metric dictionaries.
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle different formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'metrics' in data:
            return data['metrics']
        else:
            # Convert flat dict to list format
            metrics = []
            for name, values in data.items():
                if isinstance(values, list):
                    metrics.extend([{'metric_name': name, 'metric_value': v, 
                                     'timestamp': datetime.now().isoformat()} for v in values])
                else:
                    metrics.append({'metric_name': name, 'metric_value': values, 
                                   'timestamp': datetime.now().isoformat()})
            return metrics
    
    elif extension == '.jsonl':
        metrics = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        metrics.append(data)
                    except json.JSONDecodeError:
                        continue
        return metrics
    
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

def extract_metric_data(metrics: List[Dict[str, Any]], 
                       name_filter: Optional[str] = None) -> Dict[str, List[Tuple[str, float]]]:
    """Extract metric data organized by metric name.
    
    Args:
        metrics: List of metric dictionaries.
        name_filter: Optional filter for specific metric names (substring match).
        
    Returns:
        Dictionary mapping metric names to lists of (timestamp, value) tuples.
    """
    metric_data = {}
    
    for metric in metrics:
        metric_name = metric.get('metric_name', 'unknown')
        metric_value = float(metric.get('metric_value', 0.0))
        timestamp = metric.get('timestamp', datetime.now().isoformat())
        
        # Apply name filter if provided
        if name_filter is not None and name_filter.lower() not in metric_name.lower():
            continue
            
        if metric_name not in metric_data:
            metric_data[metric_name] = []
            
        metric_data[metric_name].append((timestamp, metric_value))
    
    # Sort each metric's data by timestamp
    for name in metric_data:
        metric_data[name].sort(key=lambda x: x[0])
    
    return metric_data

def plot_metrics(metric_data: Dict[str, List[Tuple[str, float]]], 
                output_path: Optional[str] = None,
                show_plot: bool = True):
    """Plot metrics over time.
    
    Args:
        metric_data: Dictionary mapping metric names to lists of (timestamp, value) tuples.
        output_path: Optional path to save the plot.
        show_plot: Whether to display the plot.
    """
    if not metric_data:
        print("No metrics to plot.")
        return
        
    # Determine number of subplots
    n_metrics = len(metric_data)
    if n_metrics <= 3:
        n_rows, n_cols = 1, n_metrics
    else:
        n_rows = (n_metrics + 1) // 2
        n_cols = 2
        
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()  # Flatten for easy indexing
    else:
        axes = axes.flatten()  # Flatten for easy indexing
        
    # Plot each metric
    for i, (name, data) in enumerate(metric_data.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Extract values and convert timestamps to sequential indices
        values = [value for _, value in data]
        indices = list(range(len(values)))
        
        # Create x-axis points
        step = max(1, len(indices) // 10)  # Show at most 10 labels
        x_tick_indices = indices[::step]
        x_tick_labels = [i+1 for i in x_tick_indices]  # 1-based step count
        
        # Plot the metric
        ax.plot(indices, values, '-o', markersize=4)
        ax.set_title(name)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis ticks
        ax.set_xticks(x_tick_indices)
        ax.set_xticklabels(x_tick_labels)
        
        # Add min, max, and last values as annotations
        if values:
            min_val = min(values)
            max_val = max(values)
            last_val = values[-1]
            
            ax.annotate(f'Min: {min_val:.4f}', xy=(0.02, 0.05), xycoords='axes fraction')
            ax.annotate(f'Max: {max_val:.4f}', xy=(0.02, 0.10), xycoords='axes fraction')
            ax.annotate(f'Last: {last_val:.4f}', xy=(0.02, 0.15), xycoords='axes fraction')
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show the plot if requested
    if show_plot:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize DAPO training metrics")
    parser.add_argument(
        '--db-path', type=str, default='data/dapo.db',
        help='Path to the DAPO database file'
    )
    parser.add_argument(
        '--input-file', type=str, default=None,
        help='Path to a JSON/JSONL metrics file (alternative to database)'
    )
    parser.add_argument(
        '--metric', type=str, default=None,
        help='Filter for specific metric name (substring match)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save the plot image'
    )
    parser.add_argument(
        '--no-show', action='store_true',
        help='Do not display the plot (useful for batch processing)'
    )
    args = parser.parse_args()
    
    # Setup logger
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger('visualize_metrics', log_dir)
    
    try:
        # Load metrics data
        if args.input_file and os.path.exists(args.input_file):
            logger.info(f"Loading metrics from file: {args.input_file}")
            metrics = load_metrics_from_file(args.input_file)
        elif os.path.exists(args.db_path):
            logger.info(f"Loading metrics from database: {args.db_path}")
            metrics = load_metrics_from_database(args.db_path, args.metric)
        else:
            logger.error("No valid metrics source provided.")
            return
        
        logger.info(f"Loaded {len(metrics)} metrics")
        
        # Extract and organize metric data
        metric_data = extract_metric_data(metrics, args.metric)
        
        # Plot metrics
        logger.info(f"Plotting {len(metric_data)} metrics")
        plot_metrics(metric_data, args.output, not args.no_show)
        
        if args.output:
            logger.info(f"Plot saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == '__main__':
    main()
