from dataclasses import dataclass


@dataclass
class ExperimentParams:
    system:         object
    logger:         object
    stop_condition: object
    visualizer:     dict
    progress:       object
    pacing:         object
    scheduler:      object
