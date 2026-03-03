import os
import json
from datetime import datetime
from dataclasses import asdict


def save_benchmark_results(results, folder="results"):
    """
    Saves benchmark results as structured JSON.
    Includes physical parameters and timestamp metadata.
    """

    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder}/benchmark_{timestamp}.json"

    if not results:
        raise ValueError("No results to save.")

    # Assume all runs used same PhysicalParams
    params = results[0].params

    data = {
        "metadata": {
            "timestamp": timestamp,
            "physical_params": asdict(params)
        },
        "results": []
    }

    for r in results:
        data["results"].append({
            "config": asdict(r.config),
            "summary": asdict(r.summary)
        })

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nResults saved to: {filename}")

    return filename


def load_benchmark_results(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    return data