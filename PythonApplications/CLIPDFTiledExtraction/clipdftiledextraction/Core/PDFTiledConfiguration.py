from pathlib import Path
from typing import List, Optional
import yaml

from pydantic import BaseModel, ConfigDict, Field


class PDFTiledConfiguration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    input_path: Path = Field(
        ...,
        description=(
            "Path to a single PDF or a directory of PDFs. Must be visible from "
            "inside the container."
        ),
    )
    output_path: Path = Field(
        ...,
        description="Directory where per-page results are written.",
    )
    pdf_dpi: int = Field(
        300,
        description=(
            "Rasterisation DPI. P&IDs benefit from 300+ to keep text legible "
            "inside individual tiles."
        ),
    )
    image_format: str = Field("PNG", description="PIL format for rasterised pages.")
    save_intermediate_images: bool = Field(
        True,
        description="Save full-page PNG and individual tile PNGs alongside results.",
    )
    skip_existing: bool = Field(
        True,
        description="Skip pages whose output JSON already exists.",
    )

    # Tiling parameters
    grid_cols: int = Field(3, description="Number of tile columns.")
    grid_rows: int = Field(3, description="Number of tile rows.")
    overlap_fraction: float = Field(
        0.15,
        description=(
            "Fractional expansion per side for each tile. 0.15 means tiles "
            "overlap ~30% with their neighbours, so edge-straddling components "
            "appear in at least two tiles."
        ),
    )

    # Per-tile prompt
    tile_prompt: str = Field(
        default=(
            "You are reading a cropped region from an engineering P&ID "
            "(Piping and Instrumentation Diagram).\n\n"
            "List ONLY the component and instrument tag numbers you can clearly "
            "read in this image. Examples of valid tags: PT-001, FCV-42A, "
            "SV-HP-3, VLOX-1, PG-103.\n\n"
            "Rules:\n"
            "- Output ONE tag per line.\n"
            "- Use the exact capitalisation visible in the diagram.\n"
            "- Do NOT invent or guess tags that are not clearly visible.\n"
            "- Do NOT include free-form descriptions, units, or pipe labels.\n"
            "- If no tags are visible, output the single word: NONE"
        ),
        description="Prompt sent to the VLM for every tile.",
    )

    # Sampling overrides for the tiled VLM pass (greedy is preferred here)
    sampling_overrides: dict = Field(
        default_factory=lambda: {"temperature": 0.0, "max_tokens": 512},
        description=(
            "SamplingParams overrides for tile inference. Greedy (temperature=0) "
            "is recommended to reduce hallucination on OCR-style prompts."
        ),
    )

    def list_input_pdfs(self) -> List[Path]:
        if self.input_path.is_file():
            if self.input_path.suffix.lower() == ".pdf":
                return [self.input_path]
            return []
        return sorted(self.input_path.glob("*.pdf"))

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PDFTiledConfiguration":
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with config_path.open() as fh:
            data = yaml.safe_load(fh) or {}
        return cls(**data)
