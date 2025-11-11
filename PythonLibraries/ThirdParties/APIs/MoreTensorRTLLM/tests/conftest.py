from pathlib import Path
import sys

more_tensorrtllm_path = Path(__file__).resolve().parents[1]
if more_tensorrtllm_path.exists():
    if str(more_tensorrtllm_path) not in sys.path:
        sys.path.append(str(more_tensorrtllm_path))
else:
    raise FileNotFoundError(
        f"MoreTensorRTLLM path {more_tensorrtllm_path} not found")

corecode_path = more_tensorrtllm_path.parents[2] / "CoreCode"
if corecode_path.exists():
    if str(corecode_path) not in sys.path:
        sys.path.append(str(corecode_path))
else:
    raise FileNotFoundError(
        f"CoreCode path {corecode_path} not found")

