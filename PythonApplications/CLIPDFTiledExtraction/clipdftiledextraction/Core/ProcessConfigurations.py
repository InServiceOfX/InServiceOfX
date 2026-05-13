"""Load all configurations needed by the tiled extraction runner."""
from pathlib import Path
from typing import Tuple

from clipdftiledextraction.Core.PDFTiledConfiguration import PDFTiledConfiguration


def load_configurations(
    pdf_config_path: Path,
    qwen_config_path: Path,
) -> Tuple[PDFTiledConfiguration, "Qwen3VLConfiguration"]:  # noqa: F821
    import sys

    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[4] / "PythonLibraries" / "HuggingFace" / "MoreMinerU"),
    )
    from moremineru.Configurations import Qwen3VLConfiguration  # type: ignore

    pdf_config = PDFTiledConfiguration.from_yaml(pdf_config_path)
    qwen_config = Qwen3VLConfiguration.from_yaml(qwen_config_path)
    return pdf_config, qwen_config
