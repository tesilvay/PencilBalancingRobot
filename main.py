# main.py


def main():
    # 1. Parse CLI
    args = parse_args()

    # 2. Resolve config (preset + overrides)
    config = CONFIG_PRESETS[args.config].copy()

    if args.experiment:
        config["experiment"] = args.experiment

    # 3. Build and run
    experiment = build_from_registry(EXPERIMENT_REGISTRY, config["experiment"])
    results = experiment.run()

    # 4. Print / save
    print_summary(results)