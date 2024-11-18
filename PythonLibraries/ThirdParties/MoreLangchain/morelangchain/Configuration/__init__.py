from pathlib import Path
import sys

if (Path(__file__).resolve().parents[4] / "HuggingFace" / \
	"MoreTransformers").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[4] / "HuggingFace" / \
	        "MoreTransformers"))
	
from morelangchain.Configuration.GroqGenerationConfiguration import (
	GroqGenerationConfiguration)