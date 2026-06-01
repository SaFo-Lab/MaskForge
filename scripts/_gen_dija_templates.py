"""One-shot: generate the 100 AdvBench dija templates and cache them so the
subsequent baseline runs don't need vLLM 8100."""
import argparse
import json
import sys
from pathlib import Path
import _bootstrap  # noqa: F401

# baselines/ holds transition_baselines.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate and cache DIJA templates for AdvBench transition baselines.",
    )
    parser.add_argument("--data", default="data/advbench.json")
    parser.add_argument("--max-goals", type=int, default=100)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    goals = json.load(open(args.data))[: args.max_goals]
    from llm import VLLMServerModel
    from transition_baselines import build_dija_templates

    attack_model = VLLMServerModel(
        model_name=args.model,
        base_url=f"http://localhost:{args.port}",
    )
    cache = build_dija_templates(goals, attack_model)
    print(f"final cache size: {len(cache)}")


if __name__ == "__main__":
    main()
