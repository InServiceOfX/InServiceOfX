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

# To obtain modules from MoreSGLang
if (Path(__file__).resolve().parents[4] / \
	"PythonLibraries" / "ThirdParties" / "MoreSGLang").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[4] / "PythonLibraries" / \
			"ThirdParties" / "MoreSGLang"))
