"""
Analysis plotting module.

Run with:
    uv run python -m royal_game_of_ur --analyse --episodes 100000 --output-dir artifacts/Run_3
    uv run python -m royal_game_of_ur --analyse --episodes 100000 --output-dir artifacts/Run_3 --parallel-sweeps

Direct module run (same plots):
    uv run python -m royal_game_of_ur.analysis --episodes 100000 --output-dir artifacts/Run_3
"""
from __future__ import annotations
 
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
 
import matplotlib.pyplot as plt
import numpy as np
 
from .environment import RoyalGameOfUrEnv
from .training import (
    TrainingResult,
    train_sarsa_lambda,
)
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _rolling_mean(values: list[float] | list[int] | np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean with no edge artifacts.
 
    For the first `window-1` points the average expands from the start,
    so the output has the same length as the input and is never distorted
    by zero-padding.
    """
    arr = np.asarray(values, dtype=float)
    cumsum = np.concatenate([[0.0], np.cumsum(arr)])
    counts = np.minimum(np.arange(1, len(arr) + 1), window)
    start_idx = np.arange(len(arr)) - counts + 1
    return (cumsum[np.arange(len(arr)) + 1] - cumsum[start_idx]) / counts
 
 
def _cumulative_win_rate(wins: list[int]) -> np.ndarray:
    arr = np.asarray(wins, dtype=float)
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)
 
 
# ---------------------------------------------------------------------------
# Plot 1 — Q-values for all 5 initial pairs (single baseline run)
# ---------------------------------------------------------------------------
 
def plot_all_initial_q(
    episodes: int,
    output_dir: Path,
    alpha: float = 0.1,
    lambda_: float = 0.8,
    seed: int = 0,
    result: TrainingResult | None = None,
) -> None:
    """Show unsmoothed + smoothed Q_t(s0,a0) for every dice value on one figure."""
    print(f"  [1/5] All-initial-Q plot  α={alpha}, λ={lambda_} …", flush=True)
    if result is None:
        env = RoyalGameOfUrEnv()
        result = train_sarsa_lambda(env, episodes=episodes, alpha=alpha,
                                    lambda_=lambda_, seed=seed)
 
    t = np.arange(1, episodes + 1)
    w = max(1, episodes // 200)           # ~0.5 % smoothing window
    plt.style.use("ggplot")
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        r"$Q_t(s_0, a_0)$ for all 5 initial state-action pairs"
        f"  (α={alpha}, λ={lambda_})",
        fontsize=12,
    )
 
    for dice, values in sorted(result.tracked_initial_values.items()):
        lbl = f"dice={dice} (pass)" if dice == 0 else f"dice={dice}"
        axes[0].plot(t, values, linewidth=0.8, alpha=0.6, label=lbl)
        axes[1].plot(t, _rolling_mean(values, w), linewidth=1.6, label=lbl)
 
    for ax, title in zip(axes, ["Unsmoothed", f"Smoothed (window={w})"]):
        ax.set_xlabel("Episode $t$")
        ax.set_ylabel(r"$Q_t(s_0, a_0)$")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=8)
 
    fig.tight_layout()
    p = output_dir / "initial_q_all_dice.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"    → {p}")
 
 
# ---------------------------------------------------------------------------
# Plot 2 — Win rate baseline (single run)
# ---------------------------------------------------------------------------
 
def plot_win_rate_baseline(
    episodes: int,
    output_dir: Path,
    alpha: float = 0.1,
    lambda_: float = 0.8,
    seed: int = 0,
    result: TrainingResult | None = None,
) -> None:
    """Cumulative and smoothed win rate for the default hyperparameters."""
    print(f"  [2/5] Win-rate baseline plot  α={alpha}, λ={lambda_} …", flush=True)
    if result is None:
        env = RoyalGameOfUrEnv()
        result = train_sarsa_lambda(env, episodes=episodes, alpha=alpha,
                                    lambda_=lambda_, seed=seed)
 
    t = np.arange(1, episodes + 1)
    w = max(1, episodes // 200)
    plt.style.use("ggplot")
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Win rate over training  (α={alpha}, λ={lambda_})",
        fontsize=12,
    )
 
    # Cumulative
    axes[0].plot(
        t,
        _cumulative_win_rate(result.wins),
        linewidth=2,
        color="tab:blue",
        label="Cumulative win rate",
    )
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=1.2,
                    label="50 % reference")
    axes[0].set_title("Cumulative win rate")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Win rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()
 
    # Smoothed per-episode
    axes[1].plot(t, _rolling_mean(result.wins, w), linewidth=2, color="tab:orange",
                 label=f"Rolling mean (window={w})")
    axes[1].axhline(0.5, color="grey", linestyle="--", linewidth=1.2,
                    label="50 % reference")
    axes[1].set_title("Smoothed per-episode win rate")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Win rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
 
    fig.tight_layout()
    p = output_dir / "win_rate_baseline.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"    → {p}")
 
 
# ---------------------------------------------------------------------------
# Plot 3 — λ sweep: Q-convergence + win rate + final-value bar
# ---------------------------------------------------------------------------
 
def plot_lambda_sweep(
    episodes: int,
    output_dir: Path,
    lambdas: tuple[float, ...] = (0.0, 0.2, 0.5, 0.8, 0.9, 0.99),
    alpha: float = 0.1,
    seed: int = 0,
) -> None:
    """Three figures that together fully answer the λ convergence question."""
    print(f"  [3/5] λ sweep  ({len(lambdas)} values × {episodes:,} episodes)", flush=True)
 
    t = np.arange(1, episodes + 1)
    w = max(1, episodes // 200)
    plt.style.use("ggplot")
 
    # Pre-allocate axes for the two multi-curve plots
    fig_q_raw,  ax_q_raw  = plt.subplots(figsize=(11, 5))
    fig_q_smo,  ax_q_smo  = plt.subplots(figsize=(11, 5))
    fig_wr,     ax_wr     = plt.subplots(figsize=(11, 5))
 
    final_q:  list[float] = []
    final_wr: list[float] = []
 
    for lam in lambdas:
        print(f"    λ={lam:.2f} …", flush=True)
        env = RoyalGameOfUrEnv()
        result = train_sarsa_lambda(env, episodes=episodes, alpha=alpha,
                                    lambda_=lam, seed=seed)
 
        q_raw = np.asarray(result.tracked_initial_values[3])
        q_smo = _rolling_mean(q_raw, w)
        wr    = _rolling_mean(result.wins, w)
 
        lbl = f"λ={lam}"
        ax_q_raw.plot(t, q_raw, linewidth=0.8, alpha=0.55, label=lbl)
        ax_q_smo.plot(t, q_smo, linewidth=1.6, label=lbl)
        ax_wr.plot(t, wr, linewidth=1.6, label=lbl)
 
        # Tail mean of last 10 % for Q and win rate as a converged-value summary.
        tail = max(1, episodes // 10)
        final_q.append(float(np.mean(q_raw[-tail:])))
        final_wr.append(float(np.mean(result.wins[-tail:])))
 
    # --- Figure A: raw Q traces ---
    ax_q_raw.set_title(
        r"$Q_t(s_0, a_0)$ for different $\lambda$  (dice=3 initial state)",
        fontsize=11,
    )
    ax_q_raw.set_xlabel("Episode $t$")
    ax_q_raw.set_ylabel(r"$Q_t(s_0, a_0)$")
    ax_q_raw.set_ylim(-0.05, 1.05)
    ax_q_raw.legend(title=r"$\lambda$", fontsize=9)
    fig_q_raw.tight_layout()
    p = output_dir / "lambda_q_raw.png"
    fig_q_raw.savefig(p, dpi=200); plt.close(fig_q_raw)
    print(f"    → {p}")
 
    # --- Figure B: smoothed Q traces (the primary required convergence plot) ---
    ax_q_smo.set_title(
        r"Smoothed $Q_t(s_0, a_0)$ for different $\lambda$  (window=" + str(w) + r" episodes)"
        "\n"
        r"$s_0$: all pieces at start, dice=3 $\;|\;$ $a_0$: move piece $0\!\to\!3$",
        fontsize=11,
    )
    ax_q_smo.set_xlabel("Episode $t$")
    ax_q_smo.set_ylabel(r"$Q_t(s_0, a_0)$")
    ax_q_smo.set_ylim(-0.05, 1.05)
    ax_q_smo.legend(title=r"$\lambda$", fontsize=9)
    fig_q_smo.tight_layout()
    p = output_dir / "lambda_q_smoothed.png"
    fig_q_smo.savefig(p, dpi=200); plt.close(fig_q_smo)
    print(f"    → {p}")
 
    # --- Figure C: smoothed win rate ---
    ax_wr.axhline(0.5, color="black", linestyle="--", linewidth=1.0,
                  label="50 % reference")
    ax_wr.set_title(
        r"Smoothed win rate for different $\lambda$  (window=" + str(w) + " episodes)",
        fontsize=11,
    )
    ax_wr.set_xlabel("Episode")
    ax_wr.set_ylabel("Win rate")
    ax_wr.set_ylim(0.0, 1.0)
    ax_wr.legend(title=r"$\lambda$", fontsize=9)
    fig_wr.tight_layout()
    p = output_dir / "lambda_win_rate.png"
    fig_wr.savefig(p, dpi=200); plt.close(fig_wr)
    print(f"    → {p}")
 
    # --- Figure D: final converged values bar chart ---
    lam_labels = [str(l) for l in lambdas]
    x = np.arange(len(lambdas))
    w_bar = 0.38
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        r"Converged values (tail mean of last 10 % of episodes) vs $\lambda$",
        fontsize=12,
    )
 
    axes[0].bar(x, final_q, width=w_bar, color="steelblue", edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels(lam_labels)
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel(r"$Q$ (tail mean, last 10 %)")
    axes[0].set_title(r"Final $Q(s_0, a_0)$ by $\lambda$")
    axes[0].set_ylim(0.0, 1.0)
 
    axes[1].bar(x, final_wr, width=w_bar, color="darkorange", edgecolor="white")
    axes[1].axhline(0.5, color="grey", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x); axes[1].set_xticklabels(lam_labels)
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel("Win rate (tail mean, last 10 %)")
    axes[1].set_title(r"Converged win rate by $\lambda$")
    axes[1].set_ylim(0.0, 1.0)
 
    fig.tight_layout()
    p = output_dir / "lambda_final_values.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"    → {p}")
 
 
# ---------------------------------------------------------------------------
# Plot 4 — α sweep: Q-convergence + win rate + final-value bar
# ---------------------------------------------------------------------------
 
def plot_alpha_sweep(
    episodes: int,
    output_dir: Path,
    alphas: tuple[float, ...] = (0.01, 0.05, 0.1, 0.3, 0.5),
    lambda_: float = 0.8,
    seed: int = 0,
) -> None:
    """Three figures for the α sensitivity analysis."""
    print(f"  [4/5] α sweep  ({len(alphas)} values × {episodes:,} episodes)", flush=True)
 
    t = np.arange(1, episodes + 1)
    w = max(1, episodes // 200)
    plt.style.use("ggplot")
 
    fig_q_smo, ax_q_smo = plt.subplots(figsize=(11, 5))
    fig_wr,    ax_wr    = plt.subplots(figsize=(11, 5))
 
    final_q:  list[float] = []
    final_wr: list[float] = []
 
    for alpha in alphas:
        print(f"    α={alpha:.3f} …", flush=True)
        env = RoyalGameOfUrEnv()
        result = train_sarsa_lambda(env, episodes=episodes, alpha=alpha,
                                    lambda_=lambda_, seed=seed)
 
        q_raw = np.asarray(result.tracked_initial_values[3])
        q_smo = _rolling_mean(q_raw, w)
        wr    = _rolling_mean(result.wins, w)
 
        lbl = f"α={alpha}"
        ax_q_smo.plot(t, q_smo, linewidth=1.6, label=lbl)
        ax_wr.plot(t, wr, linewidth=1.6, label=lbl)
 
        tail = max(1, episodes // 10)
        final_q.append(float(np.mean(q_raw[-tail:])))
        final_wr.append(float(np.mean(result.wins[-tail:])))
 
    # --- Figure A: smoothed Q convergence ---
    ax_q_smo.set_title(
        r"Smoothed $Q_t(s_0, a_0)$ for different $\alpha$  (window=" + str(w) + r" episodes)"
        f"\n(λ={lambda_})",
        fontsize=11,
    )
    ax_q_smo.set_xlabel("Episode $t$")
    ax_q_smo.set_ylabel(r"$Q_t(s_0, a_0)$")
    ax_q_smo.set_ylim(-0.05, 1.05)
    ax_q_smo.legend(title=r"$\alpha$", fontsize=9)
    fig_q_smo.tight_layout()
    p = output_dir / "alpha_q_smoothed.png"
    fig_q_smo.savefig(p, dpi=200); plt.close(fig_q_smo)
    print(f"    → {p}")
 
    # --- Figure B: smoothed win rate ---
    ax_wr.axhline(0.5, color="black", linestyle="--", linewidth=1.0,
                  label="50 % reference")
    ax_wr.set_title(
        r"Smoothed win rate for different $\alpha$  (window=" + str(w) + f" episodes, λ={lambda_})",
        fontsize=11,
    )
    ax_wr.set_xlabel("Episode")
    ax_wr.set_ylabel("Win rate")
    ax_wr.set_ylim(0.0, 1.0)
    ax_wr.legend(title=r"$\alpha$", fontsize=9)
    fig_wr.tight_layout()
    p = output_dir / "alpha_win_rate.png"
    fig_wr.savefig(p, dpi=200); plt.close(fig_wr)
    print(f"    → {p}")
 
    # --- Figure C: final converged values bar chart ---
    alpha_labels = [str(a) for a in alphas]
    x = np.arange(len(alphas))
    w_bar = 0.38
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        r"Converged values (tail mean of last 10 % of episodes) vs $\alpha$"
        f"  (λ={lambda_})",
        fontsize=12,
    )
 
    axes[0].bar(x, final_q, width=w_bar, color="steelblue", edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels(alpha_labels)
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$Q$ (tail mean, last 10 %)")
    axes[0].set_title(r"Final $Q(s_0, a_0)$ by $\alpha$")
    axes[0].set_ylim(0.0, 1.0)
 
    axes[1].bar(x, final_wr, width=w_bar, color="darkorange", edgecolor="white")
    axes[1].axhline(0.5, color="grey", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x); axes[1].set_xticklabels(alpha_labels)
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel("Win rate (tail mean, last 10 %)")
    axes[1].set_title(r"Final win rate by $\alpha$")
    axes[1].set_ylim(0.0, 1.0)
 
    fig.tight_layout()
    p = output_dir / "alpha_final_values.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"    → {p}")


# ---------------------------------------------------------------------------
# Plot 6 — Episode sweep: converged-value summary vs episode count
# ---------------------------------------------------------------------------

def plot_episode_sweep(
    output_dir: Path,
    episode_counts: tuple[int, ...],
    alpha: float = 0.1,
    lambda_: float = 0.8,
    seed: int = 0,
) -> None:
    """Summarise converged Q and win rate vs episode count."""
    counts = tuple(int(e) for e in episode_counts if int(e) > 0)
    if not counts:
        return

    print(f"  [6/6] Episode sweep  ({len(counts)} episode counts)", flush=True)
    plt.style.use("ggplot")

    final_q: list[float] = []
    final_wr: list[float] = []

    for ep in counts:
        print(f"    episodes={ep:,} …", flush=True)
        env = RoyalGameOfUrEnv()
        result = train_sarsa_lambda(
            env,
            episodes=ep,
            alpha=alpha,
            lambda_=lambda_,
            seed=seed,
        )

        q_raw = np.asarray(result.tracked_initial_values[3])
        tail = max(1, ep // 10)
        final_q.append(float(np.mean(q_raw[-tail:])))
        final_wr.append(float(np.mean(result.wins[-tail:])))

    labels = [f"{e:,}" for e in counts]
    x = np.arange(len(counts))
    w_bar = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        r"Converged values (tail mean of last 10 % of episodes) vs episode count"
        f"  (α={alpha}, λ={lambda_})",
        fontsize=12,
    )

    axes[0].bar(x, final_q, width=w_bar, color="steelblue", edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Episodes")
    axes[0].set_ylabel(r"$Q$ (tail mean, last 10 %)")
    axes[0].set_title(r"Final $Q(s_0, a_0)$ by episode count")
    axes[0].set_ylim(0.0, 1.0)

    axes[1].bar(x, final_wr, width=w_bar, color="darkorange", edgecolor="white")
    axes[1].axhline(0.5, color="grey", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Episodes")
    axes[1].set_ylabel("Win rate (tail mean, last 10 %)")
    axes[1].set_title("Final win rate by episode count")
    axes[1].set_ylim(0.0, 1.0)

    fig.tight_layout()
    p = output_dir / "episode_final_values.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"    → {p}")
 
 
# ---------------------------------------------------------------------------
# Plot 5 — Q-value dice comparison for one chosen λ (justifies pair choice)
# ---------------------------------------------------------------------------
 
def plot_dice_comparison(
    episodes: int,
    output_dir: Path,
    alpha: float = 0.1,
    lambda_: float = 0.8,
    seed: int = 0,
    result: TrainingResult | None = None,
) -> None:
    """Side-by-side: unsmoothed vs smoothed Q_t for every dice value.
    This justifies the choice of dice=3 as the representative pair and shows
    that the dice=0 (pass) Q-value stays at 0 — a useful sanity check.
    """
    print(f"  [5/5] Dice comparison plot  α={alpha}, λ={lambda_} …", flush=True)
    if result is None:
        env = RoyalGameOfUrEnv()
        result = train_sarsa_lambda(env, episodes=episodes, alpha=alpha,
                                    lambda_=lambda_, seed=seed)
 
    t = np.arange(1, episodes + 1)
    w = max(1, episodes // 200)
    plt.style.use("ggplot")
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        r"$Q_t(s_0, a_0)$ per initial dice value  —  unsmoothed (left) and smoothed (right)"
        f"\n(α={alpha}, λ={lambda_})",
        fontsize=11,
    )
 
    for dice, values in sorted(result.tracked_initial_values.items()):
        lbl = f"dice={dice} (pass)" if dice == 0 else f"dice={dice}"
        axes[0].plot(t, values,                   linewidth=0.8, alpha=0.65, label=lbl)
        axes[1].plot(t, _rolling_mean(values, w), linewidth=1.6,             label=lbl)
 
    for ax, title in zip(axes, ["Unsmoothed", f"Smoothed  (window={w})"]):
        ax.set_xlabel("Episode $t$")
        ax.set_ylabel(r"$Q_t(s_0, a_0)$")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=9)
 
    fig.tight_layout()
    p = output_dir / "dice_q_comparison.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"    → {p}")
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exercise 4: convergence analysis for SARSA(λ) on the Royal Game of Ur."
    )
    parser.add_argument("--episodes",    type=int,   default=100_000)
    parser.add_argument("--output-dir",  type=Path,  default=Path("artifacts"))
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--skip-lambda-sweep", action="store_true")
    parser.add_argument("--skip-alpha-sweep",  action="store_true")
    parser.add_argument("--skip-episode-sweep", action="store_true")
    parser.add_argument(
        "--parallel-sweeps",
        action="store_true",
        help="Run lambda and alpha sweeps in parallel when both are enabled.",
    )
    args = parser.parse_args()
 
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ep = args.episodes
 
    print(f"=== Baseline training ({ep:,} episodes) ===")
    baseline_env = RoyalGameOfUrEnv()
    baseline_result = train_sarsa_lambda(
        baseline_env,
        episodes=ep,
        seed=args.seed,
    )

    plot_all_initial_q(ep, out, seed=args.seed, result=baseline_result)
    plot_win_rate_baseline(ep, out, seed=args.seed, result=baseline_result)
 
    run_lambda = not args.skip_lambda_sweep
    run_alpha = not args.skip_alpha_sweep
    run_episode = not args.skip_episode_sweep

    episode_values = tuple(sorted({max(1_000, ep // 10), max(2_000, ep // 2), ep}))
    if run_lambda and run_alpha and args.parallel_sweeps:
        print("\n=== Running lambda/alpha sweeps in parallel ===")
        with ProcessPoolExecutor(max_workers=2) as ex:
            future_lambda = ex.submit(plot_lambda_sweep, ep, out, seed=args.seed)
            future_alpha = ex.submit(plot_alpha_sweep, ep, out, seed=args.seed)
            future_lambda.result()
            future_alpha.result()
    else:
        if run_lambda:
            plot_lambda_sweep(ep, out, seed=args.seed)
        if run_alpha:
            plot_alpha_sweep(ep, out, seed=args.seed)

    if run_episode:
        plot_episode_sweep(
            out,
            episode_values,
            seed=args.seed,
        )
 
    plot_dice_comparison(ep, out, seed=args.seed, result=baseline_result)
 
    print(f"\nDone — all figures saved to {out}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())