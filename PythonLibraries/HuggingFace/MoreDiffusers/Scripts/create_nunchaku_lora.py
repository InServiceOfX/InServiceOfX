"""
Usage example:

python create_nunchaku_lora.py --lora-path /Data1/Models/Diffusion/LoRAs/XLabs-AI/flux-RealismLora/lora.safetensors --lora-name qint-lora --quant-path /Data1/Models/Diffusion/mit-han-lab/svdq-int4-flux.1-dev/transformer_blocks.safetensors
"""

from pathlib import Path
import argparse
import sys

python_libraries_path = Path(__file__).resolve().parents[3]
corecode_directory = python_libraries_path / "CoreCode"

if not str(corecode_directory) in sys.path:
    sys.path.append(str(corecode_directory))

from corecode.Utilities import DataSubdirectories

def convert_lora_to_nunchaku(
    lora_path: str | Path,
    lora_name: str,
    quant_path: str | Path | None = None,
    output_root: str | Path | None = None,
    lora_format: str = "auto"
) -> None:
    """Convert a LoRA to Nunchaku format.
    
    Args:
        lora_path: Path to the LoRA safetensors file
        lora_name: Name for the output LoRA (without .safetensors extension)
        quant_path: Path to quantized base model (defaults to qint FLUX.1-dev in data
        dir)
        output_root: Output directory (defaults to current working directory)
        lora_format: LoRA format (auto, diffusers, comfyui, xlab)
    """
    data_sub_dirs = DataSubdirectories()
    models_dir = data_sub_dirs.ModelsDiffusion
    # Default quant_path points to FLUX.1-dev in data directory
    if quant_path is None:
        quant_path = models_dir / "mit-han-lab" / "svdq-int4-flux.1-dev" / "transformer_blocks.safetensors"
    
    # Default output_root to current working directory
    if output_root is None:
        output_root = Path.cwd()
        
    # Ensure paths are Path objects
    lora_path = Path(lora_path)
    output_root = Path(output_root)
    quant_path = Path(quant_path)
    
    # Create output directory if it doesn't exist
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "-m", "nunchaku.lora.flux.convert",
        "--quant-path", str(quant_path),
        "--lora-path", str(lora_path),
        "--output-root", str(output_root),
        "--lora-name", lora_name
    ]
    
    if lora_format != "auto":
        cmd.extend(["--lora-format", lora_format])
    
    # Execute command
    import subprocess
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(
        description="Convert LoRA to Nunchaku format for FLUX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert a local LoRA file using defaults
    %(prog)s --lora-path ./my_lora.safetensors --lora-name my-converted-lora

    # Convert from HuggingFace with custom output directory
    %(prog)s --lora-path aleksa-codes/flux-ghibsky-illustration/lora.safetensors \
            --lora-name ghibsky-int4 \
            --output-root ./converted_loras

    # Specify custom quantization path and LoRA format
    %(prog)s --lora-path ./my_lora.safetensors \
            --lora-name my-lora-int4 \
            --quant-path /path/to/transformer_blocks.safetensors \
            --lora-format diffusers

Notes:
    - Default quant-path points to nunchaku FLUX.1-dev in your models directory
    - Default output-root is the current working directory
    - LoRA formats: auto (default), diffusers, comfyui, xlab
    """)

    parser.add_argument(
        "--lora-path", 
        type=str, 
        required=True,
        help="Path to LoRA safetensors file (local path or HuggingFace model ID)"
    )
    parser.add_argument(
        "--lora-name", 
        type=str, 
        required=True,
        help="Name for output LoRA file (without .safetensors extension)"
    )
    parser.add_argument(
        "--quant-path", 
        type=str,
        help="Path to quantized base model (defaults to FLUX.1-dev in models directory)"
    )
    parser.add_argument(
        "--output-root", 
        type=str,
        help="Output directory (defaults to current working directory)"
    )
    parser.add_argument(
        "--lora-format",
        type=str,
        default="auto",
        choices=["auto", "diffusers", "comfyui", "xlab"],
        help="LoRA format to use for conversion (default: auto)"
    )
    
    args = parser.parse_args()
    
    convert_lora_to_nunchaku(
        lora_path=args.lora_path,
        lora_name=args.lora_name,
        quant_path=args.quant_path,
        output_root=args.output_root,
        lora_format=args.lora_format
    )

if __name__ == "__main__":
    main()
