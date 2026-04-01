import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import ModelParams, run_simulation, CMAS, CMA_BASE_RENTS

# OUTPUT LOCATION
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# EXPERIMENT E1 — Rent Increase Rate vs. Displacement Emergence
def experiment_e1():
    print("\n" + "="*60)
    print("  E1: Rent Increase Rate vs. Displacement Emergence")
    print("="*60)

    rent_rates = [0.02, 0.05, 0.08, 0.12, 0.16]
    rate_labels = ['2% (Low)', '5% (Moderate)', '8% (Baseline)', '12% (High)', '16% (Extreme)']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']

    all_results = {}
    for rate in rent_rates:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        params = ModelParams(rent_increase_rate=rate)
        all_results[rate] = run_simulation(params)
        print(f"  rent_increase_rate={rate:.2f} — "f"Final displaced: {all_results[rate]['pct_displaced'][-1]}%")

    steps = all_results[rent_rates[0]]['steps']

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        'E1: Emergence of Displacement — Effect of Rent Increase Rate\n'
        'CAS Housing Affordability Model | Canada (CMHC & Statistics Canada Data)',
        fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    for i, rate in enumerate(rent_rates):
        ax1.plot(steps, all_results[rate]['pct_displaced'], color=colors[i], linewidth=2.2, marker='o', markersize=4, label=rate_labels[i])
    ax1.axhline(y=30, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, label='30% threshold')
    ax1.set_title('Household Displacement Rate Over Time', fontweight='bold')
    ax1.set_xlabel('Year (Time Step)')
    ax1.set_ylabel('Displaced Households (%)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    for i, rate in enumerate(rent_rates):
        ax2.plot(steps, all_results[rate]['pct_cost_burdened'], color=colors[i], linewidth=2.2, marker='s', markersize=4, label=rate_labels[i])
    ax2.set_title('Cost-Burdened Households Over Time\n(Pre-Displacement)', fontweight='bold')
    ax2.set_xlabel('Year (Time Step)')
    ax2.set_ylabel('Cost-Burdened Households (%)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(CMAS))
    bar_width = 0.15
    for i, rate in enumerate(rent_rates):
        final_rents = [all_results[rate]['avg_rent_by_cma'][cma][-1] for cma in CMAS]
        ax3.bar(x + i * bar_width, final_rents, bar_width, label=rate_labels[i], color=colors[i], alpha=0.85)
    base_rents = [CMA_BASE_RENTS[cma] for cma in CMAS]
    ax3.bar(x + len(rent_rates) * bar_width, base_rents, bar_width, label='Baseline (Year 0)', color='#bdc3c7', alpha=0.85)
    ax3.set_title('Emergent Rent Inequality by CMA\n(Final Year)', fontweight='bold')
    ax3.set_xlabel('Canadian CMA')
    ax3.set_ylabel('Average Monthly Rent ($)')
    ax3.set_xticks(x + bar_width * 2.5)
    ax3.set_xticklabels(CMAS, rotation=20, ha='right', fontsize=8)
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3, axis='y')

    ax4 = fig.add_subplot(gs[1, 1])
    for i, rate in enumerate(rent_rates):
        ax4.plot(steps, all_results[rate]['avg_rent_burden'], color=colors[i], linewidth=2.2, marker='^', markersize=4, label=rate_labels[i])
    ax4.axhline(y=0.30, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Affordability threshold (30%)')
    ax4.set_title('Average Rent Burden Trajectory\n(Housed Households)', fontweight='bold')
    ax4.set_xlabel('Year (Time Step)')
    ax4.set_ylabel('Rent / Monthly Income Ratio')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, 'E1_rent_rate_displacement_emergence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return all_results


# EXPERIMENT E2 — Newcomer Employment Bias vs. Income Gap Emergence
def experiment_e2():
    print("\n" + "="*60)
    print("  E2: Newcomer Employment Bias vs. Income Gap Emergence")
    print("="*60)

    bias_levels = [0.00, 0.15, 0.35, 0.55, 0.75]
    bias_labels = ['0% (No bias)', '15% (Low)', '35% (Baseline)', '55% (High)', '75% (Severe)']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']

    all_results = {}
    for bias in bias_levels:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        params = ModelParams(newcomer_employment_bias=bias)
        all_results[bias] = run_simulation(params)
        final_step = -1
        gap = (all_results[bias]['pct_newcomer_displaced'][final_step] - all_results[bias]['pct_displaced'][final_step])
        print(f"  bias={bias:.2f} — "f"Newcomer disp: {all_results[bias]['pct_newcomer_displaced'][-1]}% | "f"Overall disp: {all_results[bias]['pct_displaced'][-1]}% | "f"Gap: {gap:.1f}%")

    steps = all_results[bias_levels[0]]['steps']

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        'E2: Emergence of Newcomer Displacement Gap — Effect of Employment Bias\n'
        'CAS Housing Affordability Model | Canada (CMHC & Statistics Canada Data)', fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    for i, bias in enumerate(bias_levels):
        newcomer_disp = all_results[bias]['pct_newcomer_displaced']
        overall_disp  = all_results[bias]['pct_displaced']
        ax1.plot(steps, newcomer_disp, color=colors[i], linewidth=2.2, linestyle='-', marker='o', markersize=4, label=f'Newcomer — {bias_labels[i]}')
        ax1.plot(steps, overall_disp, color=colors[i], linewidth=1.2, linestyle='--', alpha=0.6)
    ax1.set_title('Newcomer Displacement (solid) vs.\nOverall Displacement (dashed)', fontweight='bold')
    ax1.set_xlabel('Year (Time Step)')
    ax1.set_ylabel('Displaced (%)')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    for i, bias in enumerate(bias_levels):
        gap = [n - o for n, o in
               zip(all_results[bias]['pct_newcomer_displaced'],
                   all_results[bias]['pct_displaced'])]
        ax2.plot(steps, gap, color=colors[i], linewidth=2.2,
                 marker='s', markersize=4, label=bias_labels[i])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    ax2.set_title('Emergent Newcomer Displacement Gap\n(Newcomer% − Overall%)', fontweight='bold')
    ax2.set_xlabel('Year (Time Step)')
    ax2.set_ylabel('Displacement Gap (percentage points)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    for i, bias in enumerate(bias_levels):
        burden_gap = [n - o for n, o in
                      zip(all_results[bias]['pct_newcomer_burdened'],
                          all_results[bias]['pct_cost_burdened'])]
        ax3.plot(steps, burden_gap, color=colors[i], linewidth=2.2,
                 marker='^', markersize=4, label=bias_labels[i])
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    ax3.set_title('Emergent Newcomer Cost-Burden Gap\n(Newcomer Burdened% − Overall Burdened%)',
                  fontweight='bold')
    ax3.set_xlabel('Year (Time Step)')
    ax3.set_ylabel('Cost-Burden Gap (percentage points)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    final_gaps = []
    final_newcomer = []
    final_overall = []
    for bias in bias_levels:
        final_gaps.append(
            all_results[bias]['pct_newcomer_displaced'][-1] -
            all_results[bias]['pct_displaced'][-1])
        final_newcomer.append(all_results[bias]['pct_newcomer_displaced'][-1])
        final_overall.append(all_results[bias]['pct_displaced'][-1])

    x = np.arange(len(bias_levels))
    bar_width = 0.28
    ax4.bar(x - bar_width/2, final_newcomer, bar_width, label='Newcomer Displaced%', color='#e74c3c', alpha=0.85)
    ax4.bar(x + bar_width/2, final_overall,  bar_width, label='Overall Displaced%',  color='#3498db', alpha=0.85)
    ax4.set_title('Final Year: Emergent Newcomer vs.\nOverall Displacement by Bias Level', fontweight='bold')
    ax4.set_xlabel('Newcomer Employment Bias Level')
    ax4.set_ylabel('Displaced (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(bias_labels, rotation=15, ha='right', fontsize=8)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    path = os.path.join(OUTPUT_DIR, 'E2_newcomer_bias_emergence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return all_results

# EXPERIMENT E3 — Affordability Collapse Emergence
def experiment_e3():
    print("\n" + "="*60)
    print("  E3: Rent-Income Gap vs. Affordability Collapse Emergence")
    print("="*60)

    scenarios = [
        (0.03, 0.03, 'Balanced (3%/3%)'),
        (0.05, 0.03, 'Mild Gap (5%/3%)'),
        (0.08, 0.03, 'Moderate Gap (8%/3%) — Baseline'),
        (0.12, 0.02, 'Severe Gap (12%/2%)'),
        (0.16, 0.01, 'Crisis (16%/1%)'),
    ]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']

    all_results = {}
    for rent_r, inc_r, label in scenarios:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        params = ModelParams(rent_increase_rate=rent_r, income_growth_rate=inc_r)
        key = label
        all_results[key] = run_simulation(params)
        print(f"  {label:<35} — "f"Final displaced: {all_results[key]['pct_displaced'][-1]}% | "f"Avg burden: {all_results[key]['avg_rent_burden'][-1]:.3f}")

    steps = all_results[scenarios[0][2]]['steps']

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        'E3: Emergence of Affordability Collapse — Rent-Income Gap Effect\n'
        'CAS Housing Affordability Model | Canada (CMHC & Statistics Canada Data)', fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # ── Plot 1: Displacement % over time ───────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (_, _, label) in enumerate(scenarios):
        ax1.plot(steps, all_results[label]['pct_displaced'], color=colors[i], linewidth=2.2, marker='o', markersize=4, label=label)
    ax1.set_title('Household Displacement Over Time\nby Rent-Income Gap Scenario', fontweight='bold')
    ax1.set_xlabel('Year (Time Step)')
    ax1.set_ylabel('Displaced Households (%)')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Avg rent burden trajectory ─────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (_, _, label) in enumerate(scenarios):
        ax2.plot(steps, all_results[label]['avg_rent_burden'], color=colors[i], linewidth=2.2, marker='s', markersize=4, label=label)
    ax2.axhline(y=0.30, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='30% threshold')
    ax2.set_title('Average Rent Burden Trajectory\n(Housed Households)', fontweight='bold')
    ax2.set_xlabel('Year (Time Step)')
    ax2.set_ylabel('Rent / Monthly Income')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Affordability score by CMA — crisis scenario ─
    ax3 = fig.add_subplot(gs[1, 0])
    crisis_label = scenarios[-1][2]
    balanced_label = scenarios[0][2]
    for cma in CMAS:
        ax3.plot(steps,
            all_results[crisis_label]['affordability_by_cma'][cma], linewidth=2.0, marker='o', markersize=3, label=f'{cma} (Crisis)'
        )
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.6, label='Affordability = 1.0')
    ax3.set_title('CMA Affordability Scores — Crisis Scenario\n'
                  '(<1.0 = unaffordable at median income)', fontweight='bold')
    ax3.set_xlabel('Year (Time Step)')
    ax3.set_ylabel('Affordability Score')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Final displacement by scenario (summary bar) ─
    ax4 = fig.add_subplot(gs[1, 1])
    labels_short = ['Balanced', 'Mild Gap', 'Moderate\n(Baseline)', 'Severe Gap', 'Crisis']
    final_disp = [all_results[s[2]]['pct_displaced'][-1] for s in scenarios]
    newcomer_disp = [all_results[s[2]]['pct_newcomer_displaced'][-1] for s in scenarios]
    x = np.arange(len(scenarios))
    bar_width = 0.35
    ax4.bar(x - bar_width/2, final_disp, bar_width, label='Overall Displaced%',  color=colors, alpha=0.85)
    ax4.bar(x + bar_width/2, newcomer_disp, bar_width, label='Newcomer Displaced%', color=[c for c in colors], alpha=0.50, edgecolor='black', linewidth=0.8)
    ax4.set_title('Final Year Displacement by Scenario\n(Overall vs. Newcomer)', fontweight='bold')
    ax4.set_xlabel('Rent-Income Gap Scenario')
    ax4.set_ylabel('Displaced (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels_short, fontsize=8)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    path = os.path.join(OUTPUT_DIR, 'E3_affordability_collapse_emergence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return all_results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EMERGENCE EXPERIMENTS — E1, E2, E3")
    print("="*60)

    e1_results = experiment_e1()
    e2_results = experiment_e2()
    e3_results = experiment_e3()

    print("\n\n" + "="*60)
    print(" ALL EMERGENCE EXPERIMENTS COMPLETE")
    print("="*60)
    print(f" 3 PNG graphs saved to: {OUTPUT_DIR}")
    print(" Files:")
    print(" → E1_rent_rate_displacement_emergence.png")
    print(" → E2_newcomer_bias_emergence.png")
    print(" → E3_affordability_collapse_emergence.png")