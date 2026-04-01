import os
import copy
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import (ModelParams, run_simulation, build_network, step, CMAS, CMA_BASE_RENTS, CMA_MEDIAN_INCOMES)

# OUTPUT LOCATION
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# EXPERIMENT S1 — Affordability Threshold vs. Household Self-Sorting
def experiment_s1():
    print("\n" + "="*60)
    print("  S1: Affordability Threshold vs. Household Self-Sorting")
    print("="*60)

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]
    labels = ['20% (Strict)', '25% (Tight)', '30% (CMHC Standard)', '35% (Loose)', '40% (Permissive)']
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71']

    all_results = {}
    for threshold, label in zip(thresholds, labels):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        params = ModelParams(affordability_threshold=threshold)
        all_results[label] = run_simulation(params)

        # Measure self-organization: how concentrated are households
        # across CMAs in the final step
        G = all_results[label]['G']
        nbhd_ids = G.graph['neighbourhood_ids']
        hh_counts = [G.nodes[n]['num_households'] for n in nbhd_ids]
        total_housed = sum(hh_counts)
        
        conc = np.std(hh_counts) / (total_housed / len(nbhd_ids) + 1e-6)
        print(f"  threshold={threshold:.2f} ({label:<22}) — "f"Displaced: {all_results[label] ['pct_displaced'][-1]}% | "f"SO Concentration: {conc:.3f}")

    steps = all_results[labels[0]]['steps']

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        'S1: Self-Organization of Household Clusters — Affordability Threshold Effect\n'
        'CAS Housing Affordability Model | Canada (CMHC & Statistics Canada Data)', fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    for i, label in enumerate(labels):
        ax1.plot(steps, all_results[label]['pct_displaced'], color=colors[i], linewidth=2.2, marker='o', markersize=4, label=label)
    ax1.set_title('Household Displacement Rate\nby Affordability Threshold', fontweight='bold')
    ax1.set_xlabel('Year (Time Step)')
    ax1.set_ylabel('Displaced Households (%)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(CMAS))
    bar_width = 0.15
    for i, label in enumerate(labels):
        G = all_results[label]['G']
        nbhd_ids = G.graph['neighbourhood_ids']
        hh_counts = []
        for cma in CMAS:
            nbhd_id = next(n for n in nbhd_ids if G.nodes[n]['cma'] == cma)
            hh_counts.append(G.nodes[nbhd_id]['num_households'])
        ax2.bar(x + i * bar_width, hh_counts, bar_width, color=colors[i], alpha=0.85, label=label)
    ax2.set_title('Self-Organized Household Distribution\nby CMA (Final Year)', fontweight='bold')
    ax2.set_xlabel('Canadian CMA')
    ax2.set_ylabel('Number of Housed Households')
    ax2.set_xticks(x + bar_width * 2)
    ax2.set_xticklabels(CMAS, rotation=20, ha='right', fontsize=8)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = fig.add_subplot(gs[1, 0])
    for i, label in enumerate(labels):
        G = all_results[label]['G']
        nbhd_ids = G.graph['neighbourhood_ids']
        vacancy_std = []
        for t in range(len(steps)):
            vacancies = [G.nodes[n]['history_vacancy'][t]
                         for n in nbhd_ids
                         if len(G.nodes[n]['history_vacancy']) > t]
            vacancy_std.append(np.std(vacancies) if vacancies else 0)
        ax3.plot(steps[:len(vacancy_std)], vacancy_std, color=colors[i], linewidth=2.2, marker='^', markersize=4, label=label)
    ax3.set_title('Self-Organization Signal:\nVacancy Rate Divergence Across CMAs', fontweight='bold')
    ax3.set_xlabel('Year (Time Step)')
    ax3.set_ylabel('Std Dev of CMA Vacancy Rates')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    for i, label in enumerate(labels):
        ax4.plot(steps, all_results[label]['avg_rent_burden'], color=colors[i], linewidth=2.2, marker='s', markersize=4, label=label)
    for i, threshold in enumerate(thresholds):
        ax4.axhline(y=threshold, color=colors[i], linestyle=':', linewidth=1.0, alpha=0.5)
    ax4.set_title('Average Rent Burden vs.\nAffordability Threshold', fontweight='bold')
    ax4.set_xlabel('Year (Time Step)')
    ax4.set_ylabel('Avg Rent / Monthly Income')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, 'S1_affordability_threshold_self_sorting.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return all_results


# EXPERIMENT S2 — Employer Node Removal vs. Labour Network Reorganization
def experiment_s2():
    print("\n" + "="*60)
    print("  S2: Employer Node Removal vs. Labour Network Self-Reorganization")
    print("="*60)

    removal_fractions = [0.0, 0.15, 0.30, 0.50, 0.70]
    labels = ['No removal (0%)', '15% removed', '30% removed', '50% removed', '70% removed']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']

    all_results = {}

    for frac, label in zip(removal_fractions, labels):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        params = ModelParams()
        G = build_network(params)

        employer_ids = G.graph['employer_ids']

        if frac > 0:
            sorted_employers = sorted(
                employer_ids,
                key=lambda e: G.nodes[e]['wage_level'], reverse=True)
            num_to_remove = int(len(sorted_employers) * frac)
            to_remove = sorted_employers[:num_to_remove]

            for emp_id in to_remove:
                neighbours = list(G.neighbors(emp_id))
                for nbr in neighbours:
                    if G.nodes[nbr].get('type') == 'household':
                        G.nodes[nbr]['employed'] = False
                G.remove_node(emp_id)

            G.graph['employer_ids'] = [e for e in employer_ids
                                        if e not in to_remove]

        steps_list = []
        pct_displaced = []
        pct_newcomer_disp = []
        avg_burden = []
        newcomer_income = []
        local_income = []

        household_ids = G.graph['household_ids']
        neighbourhood_ids = G.graph['neighbourhood_ids']
        total_hh = len(household_ids)
        total_newcomers = sum(1 for h in household_ids
                                if G.nodes[h]['is_newcomer'])

        for hh_id in household_ids:
            hh = G.nodes[hh_id]
            if not hh['employed']:
                hh['income'] *= 0.90  # 10% income drop when unemployed
                hh['monthly_income'] = hh['income'] / 12

        for t in range(params.num_steps):
            G, step_metrics = step(G)
            steps_list.append(t + 1)
            displaced = sum(1 for h in household_ids if not G.nodes[h]['housed'])
            newcomer_disp = sum(
                1 for h in household_ids if not G.nodes[h]['housed'] and G.nodes[h]['is_newcomer']
            )
            burdens = [G.nodes[h]['rent_burden'] for h in household_ids if G.nodes[h]['housed']]

            n_income = np.mean([
                G.nodes[h]['income'] for h in household_ids if G.nodes[h]['is_newcomer']]
            )
            l_income = np.mean([
                G.nodes[h]['income'] for h in household_ids if not G.nodes[h]['is_newcomer']]
            )

            pct_displaced.append(round(displaced / total_hh * 100, 2))
            pct_newcomer_disp.append(
                round(newcomer_disp / max(1, total_newcomers) * 100, 2)
            )
            avg_burden.append(round(np.mean(burdens), 4) if burdens else 0.0)
            newcomer_income.append(round(n_income, 2))
            local_income.append(round(l_income, 2))

        all_results[label] = {
            'steps': steps_list,
            'pct_displaced': pct_displaced,
            'pct_newcomer_disp': pct_newcomer_disp,
            'avg_burden': avg_burden,
            'newcomer_income': newcomer_income,
            'local_income': local_income,
            'G': G,
        }
        income_gap = local_income[-1] - newcomer_income[-1]
        print(f"  {label:<22} — "
              f"Displaced: {pct_displaced[-1]}% | "
              f"Newcomer disp: {pct_newcomer_disp[-1]}% | "
              f"Income gap: ${income_gap:,.0f}")

    steps = all_results[labels[0]]['steps']

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        'S2: Self-Organization of Labour Network — High-Wage Employer Removal\n'
        'CAS Housing Affordability Model | Canada (CMHC & Statistics Canada Data)', fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── Plot 1: Overall vs. newcomer displacement ───────────
    ax1 = fig.add_subplot(gs[0, 0])
    for i, label in enumerate(labels):
        ax1.plot(steps, all_results[label]['pct_displaced'], color=colors[i], linewidth=2.2, linestyle='-', marker='o', markersize=4, label=f'Overall — {label}')
        ax1.plot(steps, all_results[label]['pct_newcomer_disp'], color=colors[i], linewidth=1.4, linestyle='--', alpha=0.7)
    ax1.set_title('Displacement Rate — Overall (solid)\nvs. Newcomer (dashed)', fontweight='bold')
    ax1.set_xlabel('Year (Time Step)')
    ax1.set_ylabel('Displaced (%)')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Income gap (local vs. newcomer) ─────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for i, label in enumerate(labels):
        gap = [l - n for l, n in
               zip(all_results[label]['local_income'], all_results[label]['newcomer_income'])
            ]
        ax2.plot(steps, gap, color=colors[i], linewidth=2.2, marker='s', markersize=4, label=label)
    ax2.set_title('Self-Organized Income Gap\n(Local − Newcomer Annual Income)', fontweight='bold')
    ax2.set_xlabel('Year (Time Step)')
    ax2.set_ylabel('Income Gap ($)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Newcomer income trajectory ──────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for i, label in enumerate(labels):
        ax3.plot(steps, all_results[label]['newcomer_income'], color=colors[i], linewidth=2.2, marker='^', markersize=4, label=label)
    ax3.set_title('Newcomer Average Annual Income\nUnder Different Employer Removal Levels', fontweight='bold')
    ax3.set_xlabel('Year (Time Step)')
    ax3.set_ylabel('Average Annual Income ($)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Final displacement summary bar ───────────────
    ax4 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(labels))
    bar_width = 0.35
    final_overall = [all_results[l]['pct_displaced'][-1]    for l in labels]
    final_newcomer = [all_results[l]['pct_newcomer_disp'][-1] for l in labels]
    ax4.bar(x - bar_width/2, final_overall, bar_width, label='Overall Displaced%',  color=colors, alpha=0.85)
    ax4.bar(x + bar_width/2, final_newcomer, bar_width, label='Newcomer Displaced%', color=colors, alpha=0.45, edgecolor='black', linewidth=0.8)
    ax4.set_title('Final Year: Self-Organized Displacement\nby Employer Removal Level', fontweight='bold')
    ax4.set_xlabel('Employer Removal Scenario')
    ax4.set_ylabel('Displaced (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{int(f*100)}%' for f in removal_fractions], fontsize=9)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    path = os.path.join(OUTPUT_DIR, 'S2_employer_removal_SO.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Saved: {path}")
    return all_results


# EXPERIMENT S3 — Rent Cap Policy Shock vs. Network Re-equilibration
def experiment_s3():
    print("\n" + "="*60)
    print("  S3: Rent Cap Policy Shock vs. Network Re-equilibration")
    print("="*60)

    scenarios = [
        (None, 'No rent cap (baseline)'),
        (2,    'Cap at Year 2 (early)'),
        (4,    'Cap at Year 4 (mid-early)'),
        (6,    'Cap at Year 6 (mid-late)'),
        (9,    'Cap at Year 9 (late)'),
    ]
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']

    all_results = {}

    for shock_year, label in scenarios:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        params = ModelParams()
        G = build_network(params)

        household_ids = G.graph['household_ids']
        neighbourhood_ids = G.graph['neighbourhood_ids']
        total_hh = len(household_ids)
        total_newcomers = sum(1 for h in household_ids if G.nodes[h]['is_newcomer'])

        steps_list = []
        pct_displaced = []
        pct_newcomer_disp = []
        avg_burden = []
        avg_rent_all = []
        affordability_all = []

        for t in range(params.num_steps):
            if shock_year is not None and (t + 1) >= shock_year:
                G.graph['params'] = ModelParams(rent_increase_rate=0.0)

            G, step_metrics = step(G)
            steps_list.append(t + 1)

            displaced = sum(
                1 for h in household_ids if not G.nodes[h]['housed']
            )
            newcomer_disp = sum(
                1 for h in household_ids if not G.nodes[h]['housed'] and G.nodes[h]['is_newcomer']
            )
            burdens = [G.nodes[h]['rent_burden'] for h in household_ids if G.nodes[h]['housed']]
            avg_rents = [G.nodes[n]['avg_rent'] for n in neighbourhood_ids]
            aff_scores = [G.nodes[n]['affordability_score'] for n in neighbourhood_ids]

            pct_displaced.append(round(displaced / total_hh * 100, 2))
            pct_newcomer_disp.append(
                round(newcomer_disp / max(1, total_newcomers) * 100, 2)
            )
            avg_burden.append(round(np.mean(burdens), 4) if burdens else 0.0)
            avg_rent_all.append(round(np.mean(avg_rents), 2))
            affordability_all.append(round(np.mean(aff_scores), 4))

        all_results[label] = {
            'steps': steps_list,
            'pct_displaced': pct_displaced,
            'pct_newcomer_disp': pct_newcomer_disp,
            'avg_burden': avg_burden,
            'avg_rent': avg_rent_all,
            'affordability': affordability_all,
            'shock_year': shock_year,
        }
        print(f"  {label:<32} — "
              f"Final displaced: {pct_displaced[-1]}% | "
              f"Avg rent: ${avg_rent_all[-1]:,.0f} | "
              f"Avg burden: {avg_burden[-1]:.3f}")

    steps = all_results[scenarios[0][1]]['steps']

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        'S3: Network Self-Reorganization After Rent Cap Policy Shock\n'
        'CAS Housing Affordability Model | Canada (CMHC & Statistics Canada Data)',
        fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── Plot 1: Displacement over time with shock markers ───
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (shock_year, label) in enumerate(scenarios):
        ax1.plot(steps, all_results[label]['pct_displaced'], color=colors[i], linewidth=2.2, marker='o', markersize=4, label=label)
        if shock_year is not None:
            ax1.axvline(x=shock_year, color=colors[i], linestyle=':', linewidth=1.2, alpha=0.6)
    ax1.set_title('Household Displacement After\nRent Cap Policy Shock', fontweight='bold')
    ax1.set_xlabel('Year (Time Step) — dotted line = cap applied')
    ax1.set_ylabel('Displaced Households (%)')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Average rent trajectory ─────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (shock_year, label) in enumerate(scenarios):
        ax2.plot(steps, all_results[label]['avg_rent'], color=colors[i], linewidth=2.2, marker='s', markersize=4, label=label)
        if shock_year is not None:
            ax2.axvline(x=shock_year, color=colors[i], linestyle=':', linewidth=1.2, alpha=0.6)
    ax2.set_title('Average Rent Across CMAs\nAfter Policy Shock', fontweight='bold')
    ax2.set_xlabel('Year (Time Step)')
    ax2.set_ylabel('Average Monthly Rent ($)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Network affordability score (SO metric) ─────
    ax3 = fig.add_subplot(gs[1, 0])
    for i, (shock_year, label) in enumerate(scenarios):
        ax3.plot(steps, all_results[label]['affordability'], color=colors[i], linewidth=2.2, marker='^', markersize=4, label=label)
        if shock_year is not None:
            ax3.axvline(x=shock_year, color=colors[i], linestyle=':', linewidth=1.2, alpha=0.6)
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.5, label='Affordability = 1.0')
    ax3.set_title('Network Affordability Score\n(Self-Organized Equilibrium Metric)', fontweight='bold')
    ax3.set_xlabel('Year (Time Step)')
    ax3.set_ylabel('Avg Affordability Score Across CMAs')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Newcomer displacement comparison ─────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for i, (shock_year, label) in enumerate(scenarios):
        ax4.plot(steps, all_results[label]['pct_newcomer_disp'], color=colors[i], linewidth=2.2, marker='D', markersize=4, label=label)
        if shock_year is not None:
            ax4.axvline(x=shock_year, color=colors[i], linestyle=':', linewidth=1.2, alpha=0.6)
    ax4.set_title('Newcomer Displacement After\nRent Cap Shock', fontweight='bold')
    ax4.set_xlabel('Year (Time Step)')
    ax4.set_ylabel('Newcomer Displaced (%)')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, 'S3_rent_cap_SO_reequilibration.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return all_results

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SELF-ORGANIZATION EXPERIMENTS — S1, S2, S3")
    print("="*60)

    s1_results = experiment_s1()
    s2_results = experiment_s2()
    s3_results = experiment_s3()

    print("\n\n" + "="*60)
    print("  ALL SELF-ORGANIZATION EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"  3 PNG graphs saved to: {OUTPUT_DIR}")
    print("  Files:")
    print("    → S1_affordability_threshold_self_sorting.png")
    print("    → S2_employer_removal_SO.png")
    print("    → S3_rent_cap_SO_reequilibration.png")