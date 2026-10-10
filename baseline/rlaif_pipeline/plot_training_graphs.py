import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# gcloud storage cp gs://rlaif-data-bucket-rlaif-project/output.jsonl .

# python3 plot_training_graphs.py --input ../output.jsonl --output training_curves.png --window 20

def main(args):
    scores = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    scores.append(record.get("score", record.get("best_score", 5)))
                except json.JSONDecodeError:
                    continue

    if not scores:
        print("No records found in the input file.")
        return

    print(f"Loaded {len(scores)} episodes.")

    window = min(args.window, len(scores))
    # np.convolve is a fast, 1-line way to get a moving average
    rolling_mean = np.convolve(scores, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10, 5))
    
    # Plot raw scores
    plt.plot(scores, label="Episode Score", color="#3498db", alpha=0.5, linewidth=1)
    
    # Plot rolling mean (shifted to align correctly on the x-axis)
    plt.plot(range(window - 1, len(scores)), rolling_mean, 
             label=f"Rolling Mean (w={window})", color="#e74c3c", linewidth=2)

    plt.axhline(y=5.5, color="gray", linestyle="--", label="Neutral (5.5)")

    # Formatting
    plt.ylim(0, 11)
    plt.xlabel("Episode")
    plt.ylabel("Score (1-10)")
    plt.title("RLAIF Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot => {args.output}")

    avg = sum(scores) / len(scores)
    print("-" * 30)
    print(f"Total Episodes : {len(scores)}")
    print(f"Average Score  : {avg:.2f} / 10")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to output JSONL from rlaif_pipeline.py")
    parser.add_argument("--output", default="training_curves.png", help="Output PNG path")
    parser.add_argument("--window", type=int, default=20, help="Rolling mean window size")
    args = parser.parse_args()
    main(args)