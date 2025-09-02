from pathlib import Path
import sys

from warnings import warn

# To obtain modules from moresglang
if (Path(__file__).resolve().parents[1]).exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[1]))

# To obtain modules from CoreCode
if (Path(__file__).resolve().parents[4] / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[4] / "CoreCode"))
else:
	path = Path(__file__).resolve().parents[4]
	error_message = (
		"CoreCode directory not found. Please ensure it exists at the expected "
		"location. Expected path directory: " + str(path)
	)
	raise Exception(error_message)

# To obtain modules from CommonAPI
if (Path(__file__).resolve().parents[4] / \
	"ThirdParties" / \
	"APIs" / \
	"CommonAPI").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[4] / \
			"ThirdParties" / \
			"APIs" / \
			"CommonAPI"))

moretransformers_path = Path(__file__).resolve().parents[4] / \
	"HuggingFace" / \
	"MoreTransformers"

if moretransformers_path.exists():
	if str(moretransformers_path) not in sys.path:
		sys.path.append(
			str(moretransformers_path))
else:
	warn(
		"MoreTransformers directory not found. Please ensure it exists at the "
		"expected location. Expected path directory: " + \
			str(moretransformers_path))
