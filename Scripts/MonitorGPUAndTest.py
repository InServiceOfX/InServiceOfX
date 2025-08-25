#!/usr/bin/env python3
"""
Script to run GPU monitoring alongside performance tests.
"""
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import argparse

core_code_path = Path(__file__).parents[1] / "PythonLibraries" / "CoreCode"

if core_code_path.exists() and str(core_code_path) not in sys.path:
    sys.path.append(str(core_code_path))
elif not core_code_path.exists():
    raise FileNotFoundError(f"CoreCode path not found: {core_code_path}")

from corecode.Utilities import GPUMonitor

class GPUAndTestRunner:
    def __init__(self, gpu_id: int = 0, interval: float = 1.0):
        self.gpu_id = gpu_id
        self.interval = interval
        self.monitor: Optional[GPUMonitor] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.output_dir = Path("./gpu_monitoring_results")
    
    def start_gpu_monitor(self):
        """Start GPU monitoring in a background thread."""

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the monitor
        self.monitor = GPUMonitor(gpu_id=self.gpu_id, interval=self.interval)
        
        # Start monitoring in a separate thread
        self.monitor_thread = threading.Thread(
            target=self._run_monitor,
            daemon=True  # This ensures the thread stops when main process exits
        )
        
        print(f"Starting GPU monitor for GPU {self.gpu_id}")
        print(f"Sampling interval: {self.interval} seconds")
        
        self.monitor_thread.start()
        
        # Give it a moment to start
        time.sleep(2)
        
        if not self.monitor_thread.is_alive():
            raise RuntimeError("GPU monitor thread failed to start")
        
        print(f"GPU monitor started successfully")
    
    def _run_monitor(self):
        """Internal method to run the monitor in a thread."""
        try:
            self.monitor.start_monitoring()
        except Exception as e:
            print(f"Error in GPU monitor thread: {e}")
    
    def run_test(self, test_path: str, test_name: str):
        """Run the specified test."""
        # Change to the project root directory
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            "-s",  # Show output
            test_path,
            "-k", test_name
        ]
        
        print(f"Running test: {' '.join(cmd)}")
        print("=" * 50)
        print("TEST COMPLETED")
        print("=" * 50)
        
        # Run the test
        result = subprocess.run(cmd, capture_output=False)
        
        return result.returncode
    
    def stop_gpu_monitor(self):
        """Stop the GPU monitoring."""
        if self.monitor and self.monitor_thread and self.monitor_thread.is_alive():
            print("Stopping GPU monitor...")

            # Wait for the monitor thread to finish (up to 30 seconds)
            try:
                self.monitor_thread.join(timeout=30)
                if self.monitor_thread.is_alive():
                    print("GPU monitor didn't stop gracefully, forcing shutdown...")
                    # Force stop by setting running to False
                    self.monitor.running = False
                    self.monitor_thread.join(timeout=5)
                else:
                    print("GPU monitor stopped gracefully")
            except Exception as e:
                print(f"Error stopping GPU monitor: {e}")
        else:
            print("GPU monitor not running")
        
    def save_results(self):
        """Save the monitoring results."""
        if self.monitor and self.monitor.metrics:
            try:
                print("Saving GPU monitoring results...")
                stats = self.monitor.save_results(self.output_dir)
                print(f"Results saved to: {self.output_dir.absolute()}")
                return stats
            except Exception as e:
                print(f"Error saving results: {e}")
                return None
        else:
            print("No monitoring data to save")
            return None
    
    def run(self, test_path: str, test_name: str):
        """Run GPU monitoring alongside the test."""
        try:
            # Start GPU monitoring
            self.start_gpu_monitor()
            
            # Run the test
            exit_code = self.run_test(test_path, test_name)
            
            # Stop GPU monitoring
            self.stop_gpu_monitor()
            
            # Save results
            self.save_results()
            
            print(f"\nExecution completed with exit code: {exit_code}")
            
            # Check if results were saved
            if self.output_dir.exists():
                print(f"\nGPU monitoring results saved to: {self.output_dir.absolute()}")
                for file in self.output_dir.glob("*"):
                    print(f"  - {file.name}")
            else:
                print("\nNo GPU monitoring results found!")
            
        except Exception as e:
            print(f"Error during execution: {e}")
            self.stop_gpu_monitor()
            return 1
        
        return exit_code
    
    def __del__(self):
        """Cleanup on destruction."""
        if self.monitor:
            self.monitor.running = False

def main():
    parser = argparse.ArgumentParser(
        description='Run GPU monitoring alongside pytest tests')
    parser.add_argument(
        '--gpu-id', type=int, default=0, help='GPU ID to monitor (default: 0)')
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='GPU monitoring interval in seconds (default: 1.0)')
    parser.add_argument(
        '--test-path',
        required=True,
        help='Path to the test file')
    parser.add_argument(
        '--test-name',
        required=True,
        help='Name of the test to run')
    
    args = parser.parse_args()
    
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, cleaning up...")
        sys.exit(1)
    
    # Handle signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    runner = GPUAndTestRunner(gpu_id=args.gpu_id, interval=args.interval)
    exit_code = runner.run(args.test_path, args.test_name)
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()