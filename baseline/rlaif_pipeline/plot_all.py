
import json
import argparse
import ast
import re
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================
# Example Usage:
#
# python3 plot_all.py \
#   --input RL_50_convo_log.json \
#   --output dashboard.png \
#   --window 10



def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan


def rolling(values, w):
    if len(values) < w:
        return np.array([])
    return np.convolve(values, np.ones(w) / w, mode="valid")


def extract_metrics(text):
    """
    Extract:
    PPO: {'raw_score':1, 'reward':-1.0, ....}
    """
    if "PPO:" not in text:
        return None

    try:
        match = re.search(r"PPO:\s*(\{.*\})", text)
        if not match:
            return None

        return ast.literal_eval(match.group(1))
    except:
        return None

def load_records(path):
    """
    Supports:
    1) JSON array file
    2) JSONL line-by-line file
    """

    try:
        with open(path) as f:
            data = json.load(f)

        records = []
        for row in data:
            text = row.get("textPayload", "")
            m = extract_metrics(text)
            if m:
                records.append(m)

        if records:
            return records

    except:
        pass

    # Try JSONL
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                text = row.get("textPayload", "")
                m = extract_metrics(text)
                if m:
                    records.append(m)
            except:
                continue

    return records


def save_line_plot(x, y, title, ylabel, path):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=2)
    plt.title(title)
    plt.xlabel("Update Step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_hist(values, title, xlabel, path):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=15, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(args):
    records = load_records(args.input)

    if not records:
        print("No PPO records found.")
        return

    print(f"Loaded {len(records)} PPO updates")

    os.makedirs("plots", exist_ok=True)

    n = len(records)
    w = min(args.window, n)
    x = np.arange(n)

    
    raw_score   = np.array([safe_float(r.get("raw_score")) for r in records])
    reward      = np.array([safe_float(r.get("reward")) for r in records])
    policy_loss = np.array([safe_float(r.get("policy_loss")) for r in records])
    value_loss  = np.array([safe_float(r.get("value_loss")) for r in records])
    kl_loss     = np.array([safe_float(r.get("kl_loss")) for r in records])
    total_loss  = np.array([safe_float(r.get("total_loss")) for r in records])
    ratio_mean  = np.array([safe_float(r.get("ratio_mean")) for r in records])
    ratio_max   = np.array([safe_float(r.get("ratio_max")) for r in records])
    did_update  = np.array([1 if r.get("did_update", False) else 0 for r in records])

    
    save_line_plot(x, reward,      "Reward",       "Reward", "plots/reward.png")
    save_line_plot(x, raw_score,   "Raw Score",    "Score",  "plots/raw_score.png")
    save_line_plot(x, total_loss,  "Total Loss",   "Loss",   "plots/total_loss.png")
    save_line_plot(x, policy_loss, "Policy Loss",  "Loss",   "plots/policy_loss.png")
    save_line_plot(x, value_loss,  "Value Loss",   "Loss",   "plots/value_loss.png")
    save_line_plot(x, kl_loss,     "KL Loss",      "Loss",   "plots/kl_loss.png")
    save_line_plot(x, ratio_mean,  "Ratio Mean",   "Ratio",  "plots/ratio_mean.png")
    save_line_plot(x, ratio_max,   "Ratio Max",    "Ratio",  "plots/ratio_max.png")

    save_hist(reward, "Reward Distribution", "Reward", "plots/reward_histogram.png")

    plt.figure(figsize=(10, 4))
    plt.bar(x, did_update)
    plt.title("Did Update")
    plt.xlabel("Update Step")
    plt.ylabel("0 / 1")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/did_update.png", dpi=150)
    plt.close()

    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("PPO Training Dashboard", fontsize=16, fontweight="bold")

    # 1 Reward
    ax = axes[0][0]
    ax.plot(x, reward, alpha=0.5)
    if n >= w:
        ax.plot(range(w-1, n), rolling(reward, w), lw=2)
    ax.set_title("Reward")
    ax.grid(True, alpha=0.3)

    # 2 Raw Score
    ax = axes[0][1]
    ax.plot(x, raw_score, color="green")
    if n >= w:
        ax.plot(range(w-1, n), rolling(raw_score, w), lw=2, color="red")
    ax.set_title("Raw Score")
    ax.grid(True, alpha=0.3)

    # 3 Total Loss
    ax = axes[0][2]
    ax.plot(x, total_loss, color="purple")
    ax.set_title("Total Loss")
    ax.grid(True, alpha=0.3)

    # 4 Policy Loss
    ax = axes[1][0]
    ax.plot(x, policy_loss, color="orange")
    ax.set_title("Policy Loss")
    ax.grid(True, alpha=0.3)

    # 5 Value Loss
    ax = axes[1][1]
    ax.plot(x, value_loss, color="teal")
    ax.set_title("Value Loss")
    ax.grid(True, alpha=0.3)

    # 6 KL Loss
    ax = axes[1][2]
    ax.plot(x, kl_loss, color="red")
    ax.set_title("KL Loss")
    ax.grid(True, alpha=0.3)

    # 7 Ratio stats
    ax = axes[2][0]
    ax.plot(x, ratio_mean, label="ratio_mean")
    ax.plot(x, ratio_max, label="ratio_max")
    ax.set_title("Ratio Stats")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8 Did Update
    ax = axes[2][1]
    ax.bar(x, did_update)
    ax.set_title("Did Update")
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3)

    # 9 Histogram
    ax = axes[2][2]
    ax.hist(reward, bins=15, edgecolor="black")
    ax.set_title("Reward Distribution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output, dpi=150, bbox_inches="tight")

    print(f"Saved dashboard => {args.output}")
    print("Saved individual plots => plots/")

    
    print("-" * 45)
    print(f"Updates           : {n}")
    print(f"Mean Reward       : {np.nanmean(reward):.4f}")
    print(f"Mean Total Loss   : {np.nanmean(total_loss):.4f}")
    print(f"Mean KL Loss      : {np.nanmean(kl_loss):.4f}")
    print(f"Updates Triggered : {int(np.sum(did_update))}")
    print("-" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="dashboard.png")
    parser.add_argument("--window", type=int, default=10)

    args = parser.parse_args()
    main(args)

