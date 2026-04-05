"""main.py — entry point for running experiments.

Usage:
    python main.py                          # sim preset, default system
    python main.py --preset real            # real preset
    python main.py --preset headless        # headless preset
    python main.py --system default:real    # override system inside sim preset
    python main.py --list                   # show all available presets
"""

import argparse
import sys

from src.shared import build_from_registry, resolve_preset
from src.experiment import EXPERIMENT_REGISTRY, EXPERIMENT_PRESETS
from src.experiment.metrics import Metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a control experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--preset", "-p",
        default="sim",
        choices=list(EXPERIMENT_PRESETS),
        metavar="PRESET",
        help=f"experiment preset to run (choices: {', '.join(EXPERIMENT_PRESETS)}). default: sim",
    )

    # Optional field overrides — each maps to a key in ExperimentParams.
    # Only the fields a user would ever want to swap from the CLI are listed.
    overrides = parser.add_argument_group("preset overrides (optional)")
    overrides.add_argument("--system",         metavar="TYPE:PRESET", help="override system")
    overrides.add_argument("--logger",         metavar="TYPE:PRESET", help="override logger")
    overrides.add_argument("--stop-condition", metavar="TYPE:PRESET", help="override stop condition")
    overrides.add_argument(
        "--realtime-visualizer",
        metavar="TYPE:PRESET",
        help="override realtime visualizer (e.g. sim:default, null:default)",
    )
    overrides.add_argument(
        "--offline-visualizer",
        metavar="TYPE:PRESET",
        help="override offline visualizer (e.g. 3d:default, null:default)",
    )
    overrides.add_argument("--progress",       metavar="TYPE:PRESET", help="override progress reporter")

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="list available presets and exit",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Preset helpers
# ---------------------------------------------------------------------------

def collect_overrides(args: argparse.Namespace) -> dict:
    """Return only the CLI overrides the user actually specified."""
    mapping = {
        "system":         args.system,
        "logger":         args.logger,
        "stop_condition": args.stop_condition,   # argparse maps --stop-condition → stop_condition
        "progress":            args.progress,
        "realtime_visualizer": args.realtime_visualizer,
        "offline_visualizer":  args.offline_visualizer,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def resolve_spec_string(preset_name: str, overrides: dict) -> str:
    """
    The experiment registry expects a single "type:preset" string.
    We always use the "default" type; the preset name selects the behaviour.

    Overrides are merged directly into the resolved preset dict before build,
    so we return both pieces separately for build_with_overrides().
    """
    return f"default:{preset_name}"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_experiment(spec_string: str, overrides: dict):
    """
    Build the experiment, applying any CLI overrides on top of the preset.

    If there are no overrides we delegate entirely to build_from_registry.
    If there are overrides we resolve the preset manually, patch it, then build.
    """
    registry = EXPERIMENT_REGISTRY
    type_, preset_name = spec_string.split(":", 1)
    spec = registry[type_]

    if not overrides:
        return build_from_registry(registry, spec_string)

    # Resolve preset (handles "base" inheritance), then patch.
    raw = resolve_preset(spec.Presets, preset_name)
    raw.update(overrides)

    # Re-use build_from_registry's inner loop by passing the patched dict
    # through the same resolution path.  We reconstruct a minimal spec-string
    # that points at an ephemeral preset we inject temporarily.
    _EPHEMERAL = "__cli__"
    spec.Presets[_EPHEMERAL] = raw
    try:
        experiment = build_from_registry(registry, f"{type_}:{_EPHEMERAL}")
    finally:
        spec.Presets.pop(_EPHEMERAL, None)

    return experiment


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def list_presets() -> None:
    print("Available experiment presets:\n")
    for name, preset in EXPERIMENT_PRESETS.items():
        base = f"  (extends '{preset['base']}')" if "base" in preset else ""
        print(f"  {name:<12}{base}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.list:
        list_presets()
        return 0

    spec_string = resolve_spec_string(args.preset, {})
    overrides   = collect_overrides(args)

    print(f"[main] building experiment '{args.preset}'" +
          (f" with overrides: {overrides}" if overrides else ""))

    experiment = build_experiment(spec_string, overrides)
    results    = experiment.run_experiment()

    metrics = Metrics()
    trial_metrics = [metrics.evaluate(r) for r in results]
    summary = metrics.summarize(trial_metrics)
    metrics.print_summary(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())