"""
Command-line interface for PCMDE.

Usage:
    python -m pcmde.cli.evaluate --config domains/animal/config.yaml
    python -m pcmde.cli.evaluate --config ... --ensemble /path/to/ensemble.json
    python -m pcmde.cli.evaluate --config ... --no-llm   # disable LLM stage
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the pcmde package is importable when run as 'python -m pcmde.cli.evaluate'
# or 'python path/to/pcmde/cli/evaluate.py'.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config_manager import ConfigManager
from core.pipeline_factory import build_pipeline


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PCMDE evaluation pipeline")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--ensemble", default=None, help="Override paths.ensemble_json")
    p.add_argument("--captions", default=None, help="Override paths.captions_csv")
    p.add_argument("--output", default=None, help="Override paths.output_json")
    p.add_argument("--no-llm", action="store_true", help="Disable the LLM validation stage")
    p.add_argument("--quiet", action="store_true", help="Suppress per-image output")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = ConfigManager.load(args.config)

    overrides = {
        "paths.ensemble_json": args.ensemble,
        "paths.captions_csv": args.captions,
        "paths.output_json": args.output,
    }
    if args.no_llm:
        overrides["llm.enabled"] = False
    config.override(overrides)

    ensemble_path = config.require("paths.ensemble_json")
    captions_path = config.get("paths.captions_csv")
    output_path = config.get("paths.output_json", "./pcmde_results.json")
    image_col = config.get("csv_mapping.image_col", "Image_path")
    caption_col = config.get("csv_mapping.caption_col", "Image_caption")

    runner = build_pipeline(config)

    result_package = runner.run(
        ensemble_path=ensemble_path,
        captions_path=captions_path,
        image_col=image_col,
        caption_col=caption_col,
        verbose=not args.quiet,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result_package, fh, indent=2)
    print()
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()