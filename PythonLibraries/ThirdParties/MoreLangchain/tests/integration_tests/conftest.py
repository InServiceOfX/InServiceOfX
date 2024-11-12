from pathlib import Path
import pytest
import sys

# To obtain modules from MoreLangchain
if Path(__file__).resolve().parents[2].exists():
	sys.path.append(str(Path(__file__).resolve().parents[2]))

# To obtain modules from CoreCode
if (Path(__file__).resolve().parents[4] / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[4] / "CoreCode"))

# To obtain modules from MoreTransformers
if (Path(__file__).resolve().parents[4] / "HuggingFace" / \
	"MoreTransformers").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[4] / "HuggingFace" / \
	        "MoreTransformers"))

@pytest.fixture
def more_transformers_test_data_directory():
    return Path(__file__).resolve().parents[4] / "HuggingFace" / \
		"MoreTransformers" / "tests" / "TestData"