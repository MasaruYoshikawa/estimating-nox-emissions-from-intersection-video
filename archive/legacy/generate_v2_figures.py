"""Figure generator for Phase 5 to Phase 8 versioned v2 analysis outputs.

Generates high-resolution publication-quality figures for:
- Phase 5: RQ1 Primary Two-Site Correlations & Sensitivities
- Phase 6: RQ2 Conditional Wind Stratification
- Phase 7: RQ3 Central Baseline Ablation & Paired Advantage
- Phase 8: Synthesis Inference, Multiplicity Forest Plot & Temporal Null Controls
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set global matplotlib style parameters for thesis figures
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def generate_phase5_figures(results_dir: Path, out_dir: Path) -> list[Path]:
    """Generate Phase 5 RQ1 correlation comparison & sensitivity plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # 1. RQ1 Two-Site Correlations (Figure 5A)
    rq1_csv = results_dir / "phase5_rq1_two_site_v2.csv"
    if rq1_csv.exists():
        df_rq1 = pd.read_csv(rq1_csv)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        sites = df_rq1["site"].tolist()
        y_pos = np.arange(len(sites))
        height = 0.35

        index_r = df_rq1["index_spearman_r"].values
        index_err_low = index_r - df_rq1["index_ci_low"].values
        index_err_high = df_rq1["index_ci_high"].values - index_r

        count_r = df_rq1["count_spearman_r"].values
        count_err_low = count_r - df_rq1["count_ci_low"].values
        count_err_high = df_rq1["count_ci_high"].values - count_r

        ax.barh(
            y_pos - height / 2,
            index_r,
            height,
            xerr=[index_err_low, index_err_high],
            capsize=4,
            color="#1f77b4",
            label="Microscopic Relative Emission Index ($r_{\\text{index}}$)",
            alpha=0.9,
        )
        ax.barh(
            y_pos + height / 2,
            count_r,
            height,
            xerr=[count_err_low, count_err_high],
            capsize=4,
            color="#aec7e8",
            label="Traffic Volume Baseline ($r_{\\text{count}}$)",
            alpha=0.9,
        )

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sites)
        ax.invert_yaxis()
        ax.set_xlabel("Spearman Rank Correlation ($r$) with Primary Ambient $\\text{NO}_x$")
        ax.set_title("Phase 5: Two-Site Retrospective Association (RQ1)")
        ax.grid(True, linestyle=":", alpha=0.5, axis="x")
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

        fig_path = out_dir / "fig_phase5_rq1_two_site_correlations.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close(fig)
        generated.append(fig_path)

    # 2. Lag & Background Sensitivity (Figure 5B)
    sens_csv = results_dir / "phase5_background_lag_sensitivity_v2.csv"
    if sens_csv.exists():
        df_sens = pd.read_csv(sens_csv)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

        for site, ax in zip(["Sukiyabashi", "Shinjuku"], [ax1, ax2], strict=True):
            df_site = df_sens[df_sens["site"] == site]

            df_prim = df_site[df_site["target_semantics"] == "ambient_nox_primary"].sort_values(
                "lag_hours"
            )
            df_bg = df_site[
                df_site["target_semantics"] == "ambient_minus_setagaya_background"
            ].sort_values("lag_hours")

            ax.errorbar(
                df_prim["lag_hours"],
                df_prim["spearman_r"],
                yerr=[
                    df_prim["spearman_r"] - df_prim["ci_low"],
                    df_prim["ci_high"] - df_prim["spearman_r"],
                ],
                fmt="-o",
                color="#1f77b4",
                linewidth=2,
                capsize=4,
                label="Primary Ambient $\\text{NO}_x$",
            )
            ax.errorbar(
                df_bg["lag_hours"],
                df_bg["spearman_r"],
                yerr=[
                    df_bg["spearman_r"] - df_bg["ci_low"],
                    df_bg["ci_high"] - df_bg["spearman_r"],
                ],
                fmt="--s",
                color="#ff7f0e",
                linewidth=2,
                capsize=4,
                label="Setagaya Background-Differenced $\\Delta\\text{NO}_x$",
            )

            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["0h", "+1h", "+2h"])
            ax.set_xlabel("Time Lag Window")
            ax.set_title(f"{site} Station")
            ax.grid(True, linestyle=":", alpha=0.5)

        ax1.set_ylabel("Spearman Rank Correlation ($r$)")
        ax1.legend(loc="lower left", frameon=True, facecolor="white")
        fig.suptitle("Phase 5: Background-Differencing & Lag Window Sensitivities", y=1.02)

        fig_path = out_dir / "fig_phase5_lag_background_sensitivity.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close(fig)
        generated.append(fig_path)

    return generated


def generate_phase6_figures(results_dir: Path, out_dir: Path) -> list[Path]:
    """Generate Phase 6 RQ2 conditional wind stratification figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    rq2_csv = results_dir / "phase6_rq2_wind_stratification_v2.csv"
    if not rq2_csv.exists():
        return generated

    df_rq2 = pd.read_csv(rq2_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    strata = [
        "opposing_strong",
        "opposing_weak",
        "favorable_weak",
        "favorable_moderate",
        "favorable_strong",
    ]
    strata_labels = [
        "Opposing Strong\n(<-1.0 m/s)",
        "Opposing Weak\n(-1.0 to 0 m/s)",
        "Favorable Weak\n(0 to 1.0 m/s)",
        "Favorable Mod.\n(1.0 to 2.0 m/s)",
        "Favorable Strong\n(>2.0 m/s)",
    ]

    for site, ax in zip(["Sukiyabashi", "Shinjuku"], [ax1, ax2], strict=True):
        df_site = df_rq2[df_rq2["site"] == site]

        r_movestar = df_site["movestar_r"].values
        err_m_low = r_movestar - df_site["movestar_ci_low"].values
        err_m_high = df_site["movestar_ci_high"].values - r_movestar

        r_count = df_site["count_r"].values
        err_c_low = r_count - df_site["count_ci_low"].values
        err_c_high = df_site["count_ci_high"].values - r_count

        x = np.arange(len(strata))
        width = 0.35

        ax.bar(
            x - width / 2,
            r_movestar,
            width,
            yerr=[err_m_low, err_m_high],
            capsize=3,
            color="#2ca02c",
            label="Microscopic Index ($r_{\\text{movestar}}$)",
            alpha=0.85,
        )
        ax.bar(
            x + width / 2,
            r_count,
            width,
            yerr=[err_c_low, err_c_high],
            capsize=3,
            color="#aec7e8",
            label="Count Baseline ($r_{\\text{count}}$)",
            alpha=0.85,
        )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(strata_labels, rotation=15, fontsize=8)
        ax.set_title(f"{site} Intersection Geometry")
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    ax1.set_ylabel("Spearman Rank Correlation ($r$)")
    ax1.legend(loc="upper left", frameon=True, facecolor="white")
    fig.suptitle("Phase 6: RQ2 Along-Road Wind Speed Stratification Moderation", y=1.02)

    fig_path = out_dir / "fig_phase6_rq2_wind_stratification.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)
    generated.append(fig_path)

    return generated


def generate_phase7_figures(results_dir: Path, out_dir: Path) -> list[Path]:
    """Generate Phase 7 RQ3 central baseline ablation & paired advantage figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    rq3_csv = results_dir / "phase7_rq3_baseline_ablation_v2.csv"
    if not rq3_csv.exists():
        return generated

    df_rq3 = pd.read_csv(rq3_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: Spearman r across baselines per site
    df_sites = df_rq3[df_rq3["site"].isin(["Sukiyabashi", "Shinjuku"])]
    baselines = ["movestar_nox_index", "vt_cpfm_nox", "count_only", "speed_only"]
    base_labels = [
        "MOVEStar Index\n(Primary)",
        "VT-CPFM\n(Comparator)",
        "Count Only\n(Baseline)",
        "Speed Only\n(Baseline)",
    ]

    x = np.arange(len(baselines))
    width = 0.35

    suki_r = [
        df_sites[(df_sites["site"] == "Sukiyabashi") & (df_sites["model_baseline"] == b)][
            "spearman_r"
        ].values[0]
        for b in baselines
    ]
    suki_low = [
        df_sites[(df_sites["site"] == "Sukiyabashi") & (df_sites["model_baseline"] == b)][
            "ci_low"
        ].values[0]
        for b in baselines
    ]
    suki_high = [
        df_sites[(df_sites["site"] == "Sukiyabashi") & (df_sites["model_baseline"] == b)][
            "ci_high"
        ].values[0]
        for b in baselines
    ]

    shinj_r = [
        df_sites[(df_sites["site"] == "Shinjuku") & (df_sites["model_baseline"] == b)][
            "spearman_r"
        ].values[0]
        for b in baselines
    ]
    shinj_low = [
        df_sites[(df_sites["site"] == "Shinjuku") & (df_sites["model_baseline"] == b)][
            "ci_low"
        ].values[0]
        for b in baselines
    ]
    shinj_high = [
        df_sites[(df_sites["site"] == "Shinjuku") & (df_sites["model_baseline"] == b)][
            "ci_high"
        ].values[0]
        for b in baselines
    ]

    ax1.bar(
        x - width / 2,
        suki_r,
        width,
        yerr=[np.array(suki_r) - np.array(suki_low), np.array(suki_high) - np.array(suki_r)],
        capsize=3,
        color="#1f77b4",
        label="Sukiyabashi",
        alpha=0.85,
    )
    ax1.bar(
        x + width / 2,
        shinj_r,
        width,
        yerr=[np.array(shinj_r) - np.array(shinj_low), np.array(shinj_high) - np.array(shinj_r)],
        capsize=3,
        color="#ff7f0e",
        label="Shinjuku",
        alpha=0.85,
    )

    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(base_labels, rotation=15, fontsize=8)
    ax1.set_ylabel("Spearman Rank Correlation ($r$)")
    ax1.set_title("A: Baseline Model Correlations")
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel 2: Paired Advantage Delta r vs Count Baseline
    df_primary = df_sites[df_sites["model_baseline"] == "movestar_nox_index"]
    sites = ["Sukiyabashi", "Shinjuku"]
    delta_r = df_primary["paired_delta_r_vs_count"].values
    d_low = delta_r - df_primary["paired_delta_r_ci_low"].values
    d_high = df_primary["paired_delta_r_ci_high"].values - delta_r

    y_p = np.arange(len(sites))
    ax2.barh(y_p, delta_r, height=0.4, xerr=[d_low, d_high], capsize=4, color="#2ca02c", alpha=0.85)
    ax2.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_yticks(y_p)
    ax2.set_yticklabels(sites)
    ax2.invert_yaxis()
    ax2.set_xlabel(
        "Paired Correlation Advantage ($\\Delta r = r_{\\text{index}} - r_{\\text{count}}$)"
    )
    ax2.set_title("B: Paired Advantage over Count-Only Baseline")
    ax2.grid(True, linestyle=":", alpha=0.5, axis="x")

    fig.suptitle(
        "Phase 7: Central Baseline Ablation & Paired Association Advantage (RQ3)",
        y=0.98,
        fontsize=11.5,
    )

    # Place Data Conditions Description BELOW the chart x-axes cleanly
    cond_footer = (
        "Data Conditions: Dry Weather Filter (0.0 mm Rain)  |  +1h Physical Atmospheric Transport Delay\n"
        "Effective Sample Support: Sukiyabashi N = 520 site-hours (699 total dry hours)  |  Shinjuku N = 610 site-hours (679 total dry hours)"
    )
    fig.text(
        0.5,
        0.03,
        cond_footer,
        ha="center",
        va="bottom",
        fontsize=8,
        style="italic",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#dddddd", alpha=0.95),
    )

    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    fig_path = out_dir / "fig_phase7_rq3_baseline_ablation.png"
    plt.savefig(fig_path)
    plt.close(fig)
    generated.append(fig_path)

    return generated


def generate_phase8_figures(results_dir: Path, out_dir: Path) -> list[Path]:
    """Generate Phase 8 synthesis forest plot & temporal controls figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # 1. Forest Plot of Multiplicity Adjusted Primary Hypotheses (Figure 8A)
    synth_csv = results_dir / "phase8_synthesis_inference_v2.csv"
    if synth_csv.exists():
        df_synth = pd.read_csv(synth_csv)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        hypotheses = df_synth["hypothesis_id"].tolist()
        y_pos = np.arange(len(hypotheses))

        est = df_synth["estimate"].values
        err_low = est - df_synth["ci_low"].values
        err_high = df_synth["ci_high"].values - est

        colors = [
            (
                "#1f77b4"
                if "RQ1" in h
                else "#2ca02c" if "RQ3" in h else "#d62728" if "RQ2" in h else "#7f7f7f"
            )
            for h in hypotheses
        ]

        for i in range(len(hypotheses)):
            ax.errorbar(
                est[i],
                y_pos[i],
                xerr=[[err_low[i]], [err_high[i]]],
                fmt="o",
                color=colors[i],
                ecolor=colors[i],
                markersize=6,
                capsize=4,
                linewidth=1.8,
            )
            # Annotate adjusted p-value
            p_adj = df_synth["holm_adjusted_p"].values[i]
            dec = df_synth["decision"].values[i]
            ax.text(
                0.72,
                y_pos[i],
                f"$p_{{\\text{{Holm}}}}={p_adj:.4f}$ ({dec})",
                va="center",
                fontsize=8,
            )

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(hypotheses)
        ax.invert_yaxis()
        ax.set_xlim(-0.25, 1.25)
        ax.set_xlabel("Effect Estimate ($r$ / $\\Delta r$) with 95% Bootstrap CI")
        ax.set_title(
            "Phase 8: Primary Inference Synthesis & Holm-Bonferroni Multiplicity Adjustment"
        )
        ax.grid(True, linestyle=":", alpha=0.5, axis="x")

        fig_path = out_dir / "fig_phase8_synthesis_multiplicity_forest.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close(fig)
        generated.append(fig_path)

    # 2. Subgroup & Temporal Controls (Figure 8B)
    temp_csv = results_dir / "phase8_temporal_controls_v2.csv"
    if temp_csv.exists():
        df_temp = pd.read_csv(temp_csv)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        controls = df_temp["control_type"].tolist()
        y_pos = np.arange(len(controls))

        r_mean = df_temp["null_mean_r"].values
        err_low = r_mean - df_temp["ci_low"].values
        err_high = df_temp["ci_high"].values - r_mean

        c_colors = ["#2ca02c" if p else "#d62728" for p in df_temp["dependence_preserved"]]

        for i in range(len(controls)):
            ax.errorbar(
                r_mean[i],
                y_pos[i],
                xerr=[[err_low[i]], [err_high[i]]],
                fmt="s",
                color=c_colors[i],
                ecolor=c_colors[i],
                markersize=7,
                capsize=4,
                linewidth=1.8,
            )
            role = df_temp["diagnostic_role"].values[i]
            ax.text(0.18, y_pos[i], f"{role}", va="center", fontsize=8.5, color=c_colors[i])

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([c.replace("_", " ").title() for c in controls])
        ax.invert_yaxis()
        ax.set_xlim(-0.25, 0.60)
        ax.set_xlabel("Null Spearman Rank Correlation ($r$) with 95% CI")
        ax.set_title("Phase 8: Temporal Null Control Diagnostics")
        ax.grid(True, linestyle=":", alpha=0.5, axis="x")

        fig_path = out_dir / "fig_phase8_temporal_controls.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close(fig)
        generated.append(fig_path)

    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    out_dir = args.out_dir.resolve()

    print(f"Generating figures from {results_dir} into {out_dir}...")

    p5_figs = generate_phase5_figures(results_dir, out_dir)
    print(f"Phase 5 generated: {[f.name for f in p5_figs]}")

    p6_figs = generate_phase6_figures(results_dir, out_dir)
    print(f"Phase 6 generated: {[f.name for f in p6_figs]}")

    p7_figs = generate_phase7_figures(results_dir, out_dir)
    print(f"Phase 7 generated: {[f.name for f in p7_figs]}")

    p8_figs = generate_phase8_figures(results_dir, out_dir)
    print(f"Phase 8 generated: {[f.name for f in p8_figs]}")

    # Also copy figures to thesis/figures/ for manuscript inclusion
    thesis_fig_dir = Path("thesis/figures").resolve()
    if thesis_fig_dir.exists():
        import shutil

        for fig_file in p5_figs + p6_figs + p7_figs + p8_figs:
            shutil.copy(fig_file, thesis_fig_dir / fig_file.name)
        print(f"Copied {len(p5_figs + p6_figs + p7_figs + p8_figs)} figures to {thesis_fig_dir}")

    print("All Phase 5 to Phase 8 figures generated successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
