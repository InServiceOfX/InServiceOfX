from pathlib import Path
import sys

# To obtain modules from CLIChatLocal
if Path(__file__).resolve().parents[2].exists():
	sys.path.append(str(Path(__file__).resolve().parents[2]))

# To obtain modules from CoreCode
if (Path(__file__).resolve().parents[4] / "PythonLibraries" / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[4] / "PythonLibraries" / \
			"CoreCode"))

embeddings_path = (Path(__file__).resolve().parents[4] / \
	"PythonLibraries" / "Embeddings")

if embeddings_path.exists() and str(embeddings_path) not in sys.path:
	sys.path.append(str(embeddings_path))

more_transformers_path = (Path(__file__).resolve().parents[4] / \
	"PythonLibraries" / "HuggingFace" / "MoreTransformers")

if more_transformers_path.exists():
	sys.path.append(str(more_transformers_path))

if (Path(__file__).resolve().parents[4] / \
	"PythonLibraries" / "ThirdParties" / "APIs" / "CommonAPI").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[4] / "PythonLibraries" / \
			"ThirdParties" / "APIs" / "CommonAPI"))

tools_path = (Path(__file__).resolve().parents[4] / "PythonLibraries" / "Tools")

if tools_path.exists():
	sys.path.append(str(tools_path))
