"""
Track and compare experiments across different configurations.

THE POWER MOVE:
Run eval with recursive chunking at 512 tokens.
Run eval with fixed chunking at 256 tokens.
Run eval with semantic chunking.
Compare all three side by side.
"""

import json
from pathlib import Path

class ExperimentTracker:
    """Load, compare, and visualize experiment results."""

    def __init__(self, experiments_dir: str = "./experiments"):
        self.experiments_dir = Path(experiments_dir)
        
    def list_experiments(self) -> list[str]:
        """List all saved experiment reports."""
        files = sorted(self.experiments_dir.glob("*.json"))
        return [f.stem for f in files]
    
    def load_experiment(self, name: str) -> dict:
        """Load a single experiment report."""
        path = self.experiments_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"No experiment: {name}")
        return json.loads(path.read_text()) 
    
    def compare(self, experiment_names: list[str]) -> dict:
        """Compare multiple experiments side by side.
        
        Returns a comparison dict showing each config's scores,
        the winner for each metric, and the delta between best
        and worst.
        """
        experiments = []
        for name in experiment_names:
            experiments.append(self.load_experiment(name))


        metrics = [
            "faithfulness", "answer_relevancy",
            "context_precision", "context_recall"
        ]
        comparison = {"experiments": [], "winners": {}}

        for exp in experiments:
            comparison["experiments"].append({
                "config": exp["config"],
                "scores": exp["aggregate"],
            })

        # Find the winner for each metric
        for metric in metrics:
            scores = [
                (exp["config"], exp["aggregate"][metric])
                for exp in experiments
            ]
            scores.sort(key = lambda x: x[1], reverse = True)

            best_config, best_score = scores[0]
            worst_config, worst_score = scores[-1]

            comparison["winners"][metric] = {
                "winner": best_config,
                "score": best_score,
                "delta": round(best_score - worst_score, 4)
            }

        return comparison
    
    def print_comparison(self, experiment_names: list[str]):
        """Print a readable comparison table."""
        comp = self.compare(experiment_names)

        # Header
        configs = [e["config"] for e in comp["experiments"]]
        header = f"{'METRIC':<25}" + "".join(
            f"{c:<20}" for c in configs
        )

        print(f"\n{'='*len(header)}")
        print("EXPERIMENT COMPARISON")
        print(f"{'='*len(header)}")
        print(header)
        print("-" * len(header))

        metrics = [
            "faithfulness", "answer_relevancy",
            "context_precision", "context_recall",
        ]

        for metric in metrics:
            row = f"{metric:<25}"
            winner = comp["winners"][metric]["winner"]

            for exp in comp["experiments"]:
                score = exp["scores"][metric]
                marker = " ★" if exp["config"] == winner else ""
                row += f"{score:<20.4f}{marker}"
            print(row)

        print(f"\n★ = best for that metric")

        # Show deltas
        print(f"\nBiggest gaps:")
        for metric in metrics:
            w = comp["winners"][metric]
            if w["delta"] > 0.01:
                print(f" {metric}: {w['winner']} wins "
                      f"by {w['delta']:.4f}")
