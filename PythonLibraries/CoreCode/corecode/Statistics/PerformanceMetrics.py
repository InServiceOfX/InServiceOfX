from typing import Dict, Any, Optional
import psutil, time

from warnings import warn

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warn(
        "torch is not installed. Please install it with `pip install torch` if you need it.")

class PerformanceMetrics:
    def __init__(self):
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.peak_memory_mb: Optional[float] = None
        self.gpu_memory_mb: Optional[float] = None
        
    def start_timing(self):
        """Start timing the generation process."""
        self.start_time = time.time()
        
    def record_first_token(self):
        """Record when the first token is generated."""
        if self.first_token_time is None:
            self.first_token_time = time.time()
            
    def end_timing(self):
        """End timing the generation process."""
        self.end_time = time.time()
        
    def record_memory_usage(self):
        """Record current memory usage."""
        # CPU memory
        process = psutil.Process()
        self.peak_memory_mb = process.memory_info().rss / 1024 / 1024

        if TORCH_AVAILABLE:
            # GPU memory if available
            if torch.cuda.is_available():
                self.gpu_memory_mb = torch.cuda.max_memory_allocated() / \
                    1024 / 1024

    def get_metrics(
            self,
            input_token_count: int,
            output_token_count: int,
            ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not all([self.start_time, self.end_time]):
            raise ValueError("Timing not completed")
            
        total_time = self.end_time - self.start_time
        generated_token_count = output_token_count - input_token_count
        
        metrics = {
            'total_time_seconds': total_time,
            'input_token_count': input_token_count,
            'output_token_count': output_token_count,
            'generated_token_count': generated_token_count,
            'input_tokens_per_second': input_token_count / total_time \
                if total_time > 0 else 0,
            'generated_tokens_per_second': generated_token_count / total_time \
                if total_time > 0 else 0,
            'output_tokens_per_second': output_token_count / total_time \
                if total_time > 0 else 0,
        }
        
        # Add TTFT if available
        if self.first_token_time:
            ttft = self.first_token_time - self.start_time
            metrics['time_to_first_token_seconds'] = ttft
            metrics['time_to_first_token_ms'] = ttft * 1000
            
            # Calculate inter-token latency (excluding TTFT)
            if generated_token_count > 0:
                generation_time = self.end_time - self.first_token_time
                metrics['inter_token_latency_ms'] = (
                    generation_time / generated_token_count) * 1000
                
        if self.peak_memory_mb:
            metrics['peak_memory_mb'] = self.peak_memory_mb
        if self.gpu_memory_mb:
            metrics['peak_gpu_memory_mb'] = self.gpu_memory_mb
            
        return metrics

    @staticmethod
    def summarize_metrics(metrics_list):
        total_time_avg = \
            sum(m['total_time_seconds'] for m in metrics_list) / \
                len(metrics_list)
        generated_tokens_avg = \
            sum(m['generated_token_count'] for m in metrics_list) / \
            len(metrics_list)
        tokens_per_sec_avg = \
            sum(m['generated_tokens_per_second'] for m in metrics_list) / \
            len(metrics_list)

        ttft_avg = None
        itl_avg = None
        peak_memory_avg = None
        peak_gpu_memory_avg = None

        if all('time_to_first_token_ms' in m for m in metrics_list):
            ttft_avg = \
                sum(m['time_to_first_token_ms'] for m in metrics_list) / \
                len(metrics_list)

        if all('inter_token_latency_ms' in m for m in metrics_list):
            itl_avg = \
                sum(m['inter_token_latency_ms'] for m in metrics_list) / \
                len(metrics_list)

        if all('peak_memory_mb' in m for m in metrics_list):
            peak_memory_avg = \
                sum(m['peak_memory_mb'] for m in metrics_list) / \
                    len(metrics_list)

        summary = {
            'total_generations': len(metrics_list),
            'average_total_time_seconds': total_time_avg,
            'average_generated_tokens': generated_tokens_avg,
            'average_generated_tokens_per_second': tokens_per_sec_avg,
        }

        if ttft_avg:
            summary['average_time_to_first_token_ms'] = ttft_avg
        if itl_avg:
            summary['average_inter_token_latency_ms'] = itl_avg
        if peak_memory_avg:
            summary['average_peak_memory_mb'] = peak_memory_avg

        return summary

    @staticmethod
    def print_summary(summary):
        print("\n" + "="*60)
        print("PERFORMANCE METRICS SUMMARY")
        print("="*60)
        print(f"Total generations: {summary['total_generations']}")
        print(
            f"Average total time: {summary['average_total_time_seconds']:.3f}s")
        print(
            f"Average generated tokens: {summary['average_generated_tokens']:.1f}")
        print(
            f"Average generated tokens/sec: {summary['average_generated_tokens_per_second']:.2f}")

        if 'average_time_to_first_token_ms' in summary:
            print(
                f"Average time to first token: {summary['average_time_to_first_token_ms']:.1f}ms")

        if 'average_inter_token_latency_ms' in summary:
            print(
                f"Average inter-token latency: {summary['average_inter_token_latency_ms']:.1f}ms")

        if 'average_peak_memory_mb' in summary:
            print(
                f"Average peak memory: {summary['average_peak_memory_mb']:.1f}MB")
 
        print("="*60)
        
        
