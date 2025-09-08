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
    """Enhanced performance metrics for LLM testing."""
    
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
