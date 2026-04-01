import random
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# MODEL PARAMETERS
@dataclass
class ModelParams:
    # P1: Annual rent increase rate (calibrated from CMHC rental data)
    rent_increase_rate: float = 0.08

    # P2: Annual household income growth rate (calibrated from StatCan LFS)
    income_growth_rate: float = 0.03

    # P3: Affordability threshold — CMHC core housing need definition (>30%)
    affordability_threshold: float = 0.30

    # P4: Newcomer employment disadvantage (calibrated from StatCan immigrant earnings)
    newcomer_employment_bias: float = 0.35

    # Simulation settings
    num_households: int = 150
    num_neighbourhoods: int = 6
    num_employers: int = 20
    num_steps: int = 10
    newcomer_fraction: float = 0.30

# REAL DATA CALIBRATION — CMHC 2023 & StatCan 2021 Census
CMA_BASE_RENTS = {
    "Toronto": 2100.0,
    "Vancouver": 2300.0,
    "Calgary": 1700.0,
    "Ottawa": 1650.0,
    "Montreal": 1350.0,
    "Winnipeg": 1200.0,
}

CMA_VACANCY_RATES = {
    "Toronto": 1.5,
    "Vancouver": 0.9,
    "Calgary": 2.1,
    "Ottawa": 2.2,
    "Montreal": 3.0,
    "Winnipeg": 2.8,
}

CMA_MEDIAN_INCOMES = {
    "Toronto": 85000.0,
    "Vancouver": 82000.0,
    "Calgary": 95000.0,
    "Ottawa": 98000.0,
    "Montreal": 72000.0,
    "Winnipeg": 74000.0,
}

CMAS = list(CMA_BASE_RENTS.keys())

# AGENT DEFINITIONS
def make_household(node_id: int, params: ModelParams, cma: str) -> dict:
    is_newcomer = random.random() < params.newcomer_fraction
    base_income = CMA_MEDIAN_INCOMES[cma]
    income_multiplier = (1.0 - params.newcomer_employment_bias * 0.6) \
                        if is_newcomer else 1.0
    income = base_income * income_multiplier * np.random.uniform(0.5, 1.5)

    return {
        'type': 'household',
        'node_id': node_id,
        'cma': cma,
        'is_newcomer': is_newcomer,
        'income': round(income, 2),
        'monthly_income': round(income / 12, 2),
        'rent_paid': 0.0,
        'rent_burden': 0.0,
        'housed': True,
        'displaced_count': 0,
        'employed': False,
        'history_burden': [],
    }


def make_neighbourhood(node_id: int, cma: str) -> dict:
    base_rent = CMA_BASE_RENTS[cma]
    vacancy_rate = CMA_VACANCY_RATES[cma] / 100.0
    return {
        'type': 'neighbourhood',
        'node_id': node_id,
        'cma': cma,
        'avg_rent': base_rent,
        'vacancy_rate': vacancy_rate,
        'affordability_score': 1.0,
        'num_households': 0,
        'displaced_this_step': 0,
        'history_rent': [],
        'history_vacancy': [],
    }


def make_employer(node_id: int, cma: str, params: ModelParams) -> dict:
    base_income = CMA_MEDIAN_INCOMES[cma]
    wage_level  = base_income * np.random.uniform(0.6, 1.8)
    return {
        'type': 'employer',
        'node_id': node_id,
        'cma': cma,
        'wage_level': round(wage_level, 2),
        'newcomer_bias': params.newcomer_employment_bias,
        'num_employees': 0,
    }


# NETWORK CONSTRUCTION AND DYNAMICS

def build_network(params: ModelParams) -> nx.Graph:
    G = nx.Graph()
    G.graph['params'] = params
    G.graph['step'] = 0
    node_id = 0

    # Neighbourhood nodes
    neighbourhood_ids = []
    for cma in CMAS:
        G.add_node(node_id, **make_neighbourhood(node_id, cma))
        neighbourhood_ids.append(node_id)
        node_id += 1

    # Employer nodes
    employer_ids = []
    for i in range(params.num_employers):
        cma = CMAS[i % len(CMAS)]
        G.add_node(node_id, **make_employer(node_id, cma, params))
        employer_ids.append(node_id)
        node_id += 1

    # Household nodes + edges
    household_ids = []
    weights = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
    for i in range(params.num_households):
        cma = random.choices(CMAS, weights=weights, k=1)[0]
        hh = make_household(node_id, params, cma)
        G.add_node(node_id, **hh)
        household_ids.append(node_id)

        # Tenancy edge
        nbhd_id = next(
            n for n in neighbourhood_ids if G.nodes[n]['cma'] == cma
        )
        G.add_edge(node_id, nbhd_id, edge_type='tenancy', rent=CMA_BASE_RENTS[cma])
        G.nodes[nbhd_id]['num_households'] += 1

        # Employment edge — newcomers biased toward lower-wage employers
        cma_employers = [e for e in employer_ids if G.nodes[e]['cma'] == cma] \
                        or employer_ids
        if hh['is_newcomer']:
            wages = [G.nodes[e]['wage_level'] for e in cma_employers]
            inv_weights = [1.0 / (w + 1) for w in wages]
            employer_id = random.choices(cma_employers, weights=inv_weights, k=1)[0]
        else:
            employer_id = random.choice(cma_employers)

        G.add_edge(node_id, employer_id, edge_type='employment')
        G.nodes[node_id]['employed'] = True
        G.nodes[employer_id]['num_employees'] += 1
        node_id += 1

    G.graph['household_ids'] = household_ids
    G.graph['neighbourhood_ids'] = neighbourhood_ids
    G.graph['employer_ids'] = employer_ids
    return G

def step(G: nx.Graph) -> tuple:
    params = G.graph['params']
    G.graph['step'] += 1

    neighbourhood_ids = G.graph['neighbourhood_ids']
    household_ids = G.graph['household_ids']

    # ── 1. Update rents ────────────────────────────────────
    for nbhd_id in neighbourhood_ids:
        nbhd = G.nodes[nbhd_id]
        vacancy_multiplier = 1.0 + max(0, (0.03 - nbhd['vacancy_rate']) * 2)
        nbhd['avg_rent'] = round(
            nbhd['avg_rent'] * (1 + params.rent_increase_rate * vacancy_multiplier), 2)
        nbhd['vacancy_rate'] = max(0.005, nbhd['vacancy_rate'] * 0.97)
        nbhd['displaced_this_step'] = 0
        nbhd['history_rent'].append(nbhd['avg_rent'])
        nbhd['history_vacancy'].append(nbhd['vacancy_rate'])

    # ── 2. Update household income and rent burden ─────────
    for hh_id in household_ids:
        hh = G.nodes[hh_id]
        if not hh['housed']:
            continue
        growth = params.income_growth_rate
        if hh['is_newcomer']:
            growth *= (1.0 - params.newcomer_employment_bias * 0.4)
        hh['income'] *= (1 + growth)
        hh['monthly_income'] = hh['income'] / 12

        nbhd_id = _get_neighbourhood(G, hh_id)
        if nbhd_id is None:
            continue
        current_rent = G.nodes[nbhd_id]['avg_rent']
        G[hh_id][nbhd_id]['rent'] = current_rent
        hh['rent_paid'] = current_rent
        hh['rent_burden'] = current_rent / hh['monthly_income'] \
                               if hh['monthly_income'] > 0 else 1.0
        hh['history_burden'].append(round(hh['rent_burden'], 4))

    # ── 3. Count cost-burdened BEFORE displacement ─────────
    pre_disp_burdened = sum(
        1 for h in household_ids
        if G.nodes[h]['housed']
        and G.nodes[h]['rent_burden']
        > params.affordability_threshold
    )
    pre_disp_newcomer_b = sum(
        1 for h in household_ids
        if G.nodes[h]['housed']
        and G.nodes[h]['is_newcomer']
        and G.nodes[h]['rent_burden']
        > params.affordability_threshold
    )

    # ── 4. Displacement ────────────────────────────────────
    for hh_id in household_ids:
        hh = G.nodes[hh_id]
        if not hh['housed']:
            continue
        if hh['rent_burden'] > params.affordability_threshold:
            relocated = _attempt_relocation(G, hh_id, params)
            if not relocated:
                hh['housed'] = False
                hh['displaced_count'] += 1
                nbhd_id = _get_neighbourhood(G, hh_id)
                if nbhd_id:
                    G.nodes[nbhd_id]['num_households'] = max(
                        0, G.nodes[nbhd_id]['num_households'] - 1
                    )
                    G.nodes[nbhd_id]['displaced_this_step'] += 1
                    G.nodes[nbhd_id]['vacancy_rate'] = min(
                        1.0, G.nodes[nbhd_id]['vacancy_rate'] + 0.002
                    )

    # ── 5. Update neighbourhood affordability scores ───────
    for nbhd_id in neighbourhood_ids:
        nbhd = G.nodes[nbhd_id]
        monthly_median = CMA_MEDIAN_INCOMES[nbhd['cma']] / 12
        nbhd['affordability_score'] = round(
            (params.affordability_threshold * monthly_median) / nbhd['avg_rent'], 3
        )

    step_metrics = {
        'pre_disp_burdened': pre_disp_burdened,
        'pre_disp_newcomer_b': pre_disp_newcomer_b,
    }
    return G, step_metrics

def _get_neighbourhood(G: nx.Graph, hh_id: int) -> Optional[int]:
    for nbr in G.neighbors(hh_id):
        if G.nodes[nbr].get('type') == 'neighbourhood':
            return nbr
    return None

def _attempt_relocation(G: nx.Graph, hh_id: int, params: ModelParams) -> bool:
    hh = G.nodes[hh_id]
    current_nbhd = _get_neighbourhood(G, hh_id)
    affordable = []

    for nbhd_id in G.graph['neighbourhood_ids']:
        if nbhd_id == current_nbhd:
            continue
        nbhd = G.nodes[nbhd_id]
        rent_ratio = nbhd['avg_rent'] / hh['monthly_income'] \
                     if hh['monthly_income'] > 0 else 1.0
        if rent_ratio <= params.affordability_threshold \
                and nbhd['vacancy_rate'] > 0.01:
            affordable.append((nbhd_id, rent_ratio))

    if not affordable:
        return False

    affordable.sort(key=lambda x: x[1])
    new_nbhd_id = affordable[0][0]

    if current_nbhd is not None and G.has_edge(hh_id, current_nbhd):
        G.remove_edge(hh_id, current_nbhd)
        G.nodes[current_nbhd]['num_households'] = max(
            0, G.nodes[current_nbhd]['num_households'] - 1)

    G.add_edge(
        hh_id, new_nbhd_id, edge_type='tenancy', rent=G.nodes[new_nbhd_id]['avg_rent']
    )
    G.nodes[new_nbhd_id]['num_households'] += 1
    hh['cma'] = G.nodes[new_nbhd_id]['cma']
    hh['rent_paid'] = G.nodes[new_nbhd_id]['avg_rent']
    hh['rent_burden'] = hh['rent_paid'] / hh['monthly_income'] \
                        if hh['monthly_income'] > 0 else 1.0
    return True

# RUN SIMULATION
def run_simulation(params: ModelParams) -> dict:
    G = build_network(params)
    neighbourhood_ids = G.graph['neighbourhood_ids']
    household_ids = G.graph['household_ids']

    results = {
        'params': params,
        'steps': [],
        'avg_rent_by_cma': {cma: [] for cma in CMAS},
        'vacancy_by_cma': {cma: [] for cma in CMAS},
        'affordability_by_cma': {cma: [] for cma in CMAS},
        'pct_displaced': [],
        'pct_cost_burdened': [],
        'pct_newcomer_displaced': [],
        'pct_newcomer_burdened': [],
        'avg_rent_burden': [],
        'G': G,
    }

    for t in range(params.num_steps):
        G, step_metrics = step(G)
        results['steps'].append(t + 1)
        total_hh = len(household_ids)
        total_newcomers = sum(
            1 for h in household_ids if G.nodes[h]['is_newcomer']
        )

        # Neighbourhood metrics
        for nbhd_id in neighbourhood_ids:
            nbhd = G.nodes[nbhd_id]
            cma = nbhd['cma']
            results['avg_rent_by_cma'][cma].append(nbhd['avg_rent'])
            results['vacancy_by_cma'][cma].append(nbhd['vacancy_rate'])
            results['affordability_by_cma'][cma].append(nbhd['affordability_score'])

        # Household metrics
        displaced = sum(
            1 for h in household_ids if not G.nodes[h]['housed']
        )
        newcomer_disp = sum(
            1 for h in household_ids if not G.nodes[h]['housed'] and G.nodes[h]['is_newcomer']
            )
        burdens = [G.nodes[h]['rent_burden'] for h in household_ids if G.nodes[h]['housed']]

        results['pct_displaced'].append(
            round(displaced / total_hh * 100, 2)
        )
        results['pct_cost_burdened'].append(
            round(step_metrics['pre_disp_burdened'] / total_hh * 100, 2)
        )
        results['pct_newcomer_displaced'].append(
            round(newcomer_disp / max(1, total_newcomers) * 100, 2)
        )
        results['pct_newcomer_burdened'].append(
            round(step_metrics['pre_disp_newcomer_b'] / max(1, total_newcomers) * 100, 2)
        )
        results['avg_rent_burden'].append(
            round(np.mean(burdens), 4) if burdens else 0.0
        )

    results['G'] = G
    return results

# SMOKE TEST
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MODEL SMOKE TEST — Default Parameters")
    print("="*60)

    params  = ModelParams()
    results = run_simulation(params)
    G = results['G']

    print(f"\n  Network: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    print(f"\n  {'Step':<6} {'Displaced%':<14} {'CostBurdened%':<16} "
          f"{'NewcomerDisp%':<16} {'NewcomerBurd%':<16} {'AvgBurden'}")
    print(f"  {'-'*80}")
    for i, t in enumerate(results['steps']):
        print(f"  {t:<6} "
              f"{results['pct_displaced'][i]:<14} "
              f"{results['pct_cost_burdened'][i]:<16} "
              f"{results['pct_newcomer_displaced'][i]:<16} "
              f"{results['pct_newcomer_burdened'][i]:<16} "
              f"{results['avg_rent_burden'][i]}")

    print(f"\n  Final avg rent by CMA:")
    for cma in CMAS:
        s = CMA_BASE_RENTS[cma]
        f = results['avg_rent_by_cma'][cma][-1]
        print(f"    {cma:<12}: ${s:,.0f} → ${f:,.0f}")

    print(f"\n model.py smoke test passed. \n")