from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time

@dataclass
class GPUMetrics:
    timestamp: float
    gpu_id: int
    memory_used_mib: int
    memory_total_mib: int
    gpu_utilization: int
    temperature: int
    power_usage_w: int
    power_cap_w: int

@dataclass
class GPUStatistics:
    gpu_id: int
    total_samples: int
    max_memory_used_mib: int
    avg_memory_used_mib: float
    max_gpu_utilization: int
    avg_gpu_utilization: float
    max_temperature: int
    avg_temperature: float
    max_power_usage_w: int
    avg_power_usage_w: float
    monitoring_duration_seconds: float
    memory_usage_percentiles: dict

class GPUMonitor:
    def __init__(self, gpu_id: int = 0, interval: float = 5.0):
        self.gpu_id = gpu_id
        self.interval = interval
        self.metrics: List[GPUMetrics] = []
        self.running = False
        
    def parse_nvidia_smi_output(self, output: str) -> Optional[GPUMetrics]:
        """Parse nvidia-smi output to extract GPU metrics."""
        lines = output.strip().split('\n')
        
        # Debug: Print first few lines to see the format
        if not self.metrics:  # Only print on first call
            print("=== Debug: nvidia-smi output format ===")
            for i, line in enumerate(lines[:10]):
                print(f"Line {i}: {repr(line)}")
            print("======================================")
        
        # Look for the GPU line that starts with the GPU ID
        # The format is typically: | 0  NVIDIA GeForce GTX 980 Ti      Off | ...
        gpu_line_index = None
        for i, line in enumerate(lines):
            # Look for line that starts with | followed by the GPU ID
            if re.match(rf'^\|\s*{self.gpu_id}\s+', line):
                gpu_line_index = i
                break
        
        if gpu_line_index is None:
            print(f"Could not find GPU {self.gpu_id} in nvidia-smi output")
            return None
        
        # The metrics are typically on the next line after the GPU header
        if gpu_line_index + 1 >= len(lines):
            print("No metrics line found after GPU line")
            return None
        
        metrics_line = lines[gpu_line_index + 1]
        
        # Debug: Print the metrics line
        if not self.metrics:  # Only print on first call
            print(f"Metrics line: {repr(metrics_line)}")
        
        # Parse memory usage: "1909MiB / 6144MiB"
        memory_match = re.search(r'(\d+)MiB\s*/\s*(\d+)MiB', metrics_line)
        if not memory_match:
            print(f"Could not parse memory usage from: {metrics_line}")
            return None
        
        memory_used = int(memory_match.group(1))
        memory_total = int(memory_match.group(2))
        
        # Parse GPU utilization: "5%"
        util_match = re.search(r'(\d+)%', metrics_line)
        gpu_util = int(util_match.group(1)) if util_match else 0
        
        # Parse temperature: "67C"
        temp_match = re.search(r'(\d+)C', metrics_line)
        temperature = int(temp_match.group(1)) if temp_match else 0
        
        # Parse power usage: "80W / 275W"
        power_match = re.search(r'(\d+)W\s*/\s*(\d+)W', metrics_line)
        power_used = int(power_match.group(1)) if power_match else 0
        power_cap = int(power_match.group(2)) if power_match else 0
        
        return GPUMetrics(
            timestamp=time.time(),
            gpu_id=self.gpu_id,
            memory_used_mib=memory_used,
            memory_total_mib=memory_total,
            gpu_utilization=gpu_util,
            temperature=temperature,
            power_usage_w=power_used,
            power_cap_w=power_cap
        )
    
    def get_gpu_metrics(self) -> Optional[GPUMetrics]:
        """Get current GPU metrics using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return self.parse_nvidia_smi_output(result.stdout)
            else:
                print(f"nvidia-smi failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("nvidia-smi command timed out")
            return None
        except FileNotFoundError:
            print("nvidia-smi not found. Make sure NVIDIA drivers are installed.")
            return None
        except Exception as e:
            print(f"Error running nvidia-smi: {e}")
            return None
    
    def start_monitoring(self):
        """Start monitoring GPU metrics."""
        self.running = True
        start_time = time.time()
        
        print(f"Starting GPU monitoring for GPU {self.gpu_id}")
        print(f"Sampling interval: {self.interval} seconds")
        print("Press Ctrl+C to stop monitoring")

        try:
            while self.running:
                metrics = self.get_gpu_metrics()
                if metrics:
                    self.metrics.append(metrics)
                else:
                    print(f"Failed to get metrics for GPU {self.gpu_id}")
                
                # Sleep in smaller chunks to check stop file more frequently
                sleep_time = min(self.interval, 0.5)  # Check at least every 0.5 seconds
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping GPU monitoring...")
            self.running = False


    def calculate_statistics(self) -> GPUStatistics:
        """Calculate statistics from collected metrics."""
        if not self.metrics:
            raise ValueError("No metrics collected")
        
        memory_used_values = [m.memory_used_mib for m in self.metrics]
        gpu_util_values = [m.gpu_utilization for m in self.metrics]
        temp_values = [m.temperature for m in self.metrics]
        power_values = [m.power_usage_w for m in self.metrics]

        monitoring_duration = \
            self.metrics[-1].timestamp - self.metrics[0].timestamp

        # Calculate percentiles for memory usage
        memory_used_values.sort()
        percentiles = {
            'p50': statistics.median(memory_used_values),
            'p90': memory_used_values[int(0.9 * len(memory_used_values))],
            'p95': memory_used_values[int(0.95 * len(memory_used_values))],
            'p99': memory_used_values[int(0.99 * len(memory_used_values))]
        }

        return GPUStatistics(
            gpu_id=self.gpu_id,
            total_samples=len(self.metrics),
            max_memory_used_mib=max(memory_used_values),
            avg_memory_used_mib=statistics.mean(memory_used_values),
            max_gpu_utilization=max(gpu_util_values),
            avg_gpu_utilization=statistics.mean(gpu_util_values),
            max_temperature=max(temp_values),
            avg_temperature=statistics.mean(temp_values),
            max_power_usage_w=max(power_values),
            avg_power_usage_w=statistics.mean(power_values),
            monitoring_duration_seconds=monitoring_duration,
            memory_usage_percentiles=percentiles
        )

    def save_results(self, output_dir: Path):
        """Save monitoring results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw metrics
        metrics_file = output_dir / f"gpu_{self.gpu_id}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=2)

        # Save statistics
        stats = self.calculate_statistics()
        stats_file = output_dir / f"gpu_{self.gpu_id}_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(asdict(stats), f, indent=2)

        # Print summary
        print(f"\n=== GPU {self.gpu_id} Monitoring Summary ===")
        print(f"Monitoring duration: {stats.monitoring_duration_seconds:.2f} seconds")
        print(f"Total samples: {stats.total_samples}")
        print(f"Memory usage:")
        print(f"  Max: {stats.max_memory_used_mib} MiB")
        print(f"  Average: {stats.avg_memory_used_mib:.1f} MiB")
        print(f"  P50: {stats.memory_usage_percentiles['p50']} MiB")
        print(f"  P90: {stats.memory_usage_percentiles['p90']} MiB")
        print(f"  P95: {stats.memory_usage_percentiles['p95']} MiB")
        print(f"  P99: {stats.memory_usage_percentiles['p99']} MiB")
        print(f"GPU utilization:")
        print(f"  Max: {stats.max_gpu_utilization}%")
        print(f"  Average: {stats.avg_gpu_utilization:.1f}%")
        print(f"Temperature:")
        print(f"  Max: {stats.max_temperature}°C")
        print(f"  Average: {stats.avg_temperature:.1f}°C")
        print(f"Power usage:")
        print(f"  Max: {stats.max_power_usage_w}W")
        print(f"  Average: {stats.avg_power_usage_w:.1f}W")
        print(f"\nResults saved to:")
        print(f"  Metrics: {metrics_file}")
        print(f"  Statistics: {stats_file}")

        return stats

def main():
    parser = argparse.ArgumentParser(
        description='Monitor GPU usage with nvidia-smi')
    parser.add_argument(
        '--gpu-id', type=int, default=0, help='GPU ID to monitor (default: 0)')
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Sampling interval in seconds (default: 1.0)')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./gpu_monitoring_results', 
        help='Output directory for results (default: ./gpu_monitoring_results)')
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(gpu_id=args.gpu_id, interval=args.interval)
    
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, stopping monitoring and saving results...")
        monitor.running = False
    
    # Handle multiple signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        monitor.start_monitoring()
        monitor.save_results(Path(args.output_dir))
    except Exception as e:
        print(f"Error during monitoring: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
