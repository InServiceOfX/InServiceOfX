"""Start the P&ID Viewer web server."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pidviewer.ViewerConfiguration import ViewerConfiguration
from pidviewer.ViewerServer import create_app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P&ID extraction viewer — FastAPI/uvicorn server",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "Configurations"
        / "viewer_configuration.yml",
        help="Path to viewer_configuration.yml (default: Configurations/viewer_configuration.yml)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override host from config (default: use config value)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port from config (default: use config value)",
    )
    parser.add_argument(
        "--currentpath",
        action="store_true",
        help="Resolve --config relative to the current working directory",
    )
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = _parse_args()

    config_path = args.config
    if args.currentpath:
        config_path = Path.cwd() / config_path.name

    configuration = ViewerConfiguration.from_yaml(config_path)

    host = args.host or configuration.host
    port = args.port or configuration.port

    app = create_app(configuration)
    print(f"Starting P&ID Viewer on http://{host}:{port}")
    print(f"  mineru_output_path: {configuration.mineru_output_path}")
    if configuration.qwen3vl_output_path:
        print(f"  qwen3vl_output_path: {configuration.qwen3vl_output_path}")
    if configuration.colqwen_index_path:
        print(f"  colqwen_index_path: {configuration.colqwen_index_path}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
