import os
import sys
import time
import random
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# OUTPUT LOCATION
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HELPERS
def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

# Smoke test to verify model runs and produces expected outputs
def run_model_test():
    section("STEP 1 — Model Smoke Test")
    from model import ModelParams, run_simulation, CMAS, CMA_BASE_RENTS
    params = ModelParams()
    results = run_simulation(params)
    G = results['G']
    print(f"\n  Network : {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    print(f"  Households : {len(G.graph['household_ids'])}")
    print(f"  Neighbourhoods : {len(G.graph['neighbourhood_ids'])}")
    print(f"  Employers : {len(G.graph['employer_ids'])}")
    print(f"\n  Default run — final year summary:")
    print(f"    Displaced : {results['pct_displaced'][-1]}%")
    print(f"    Cost-burdened : {results['pct_cost_burdened'][-1]}%")
    print(f"    Newcomer disp. : {results['pct_newcomer_displaced'][-1]}%")
    print(f"    Avg rent burden : {results['avg_rent_burden'][-1]:.3f}")
    print(f"\n  Final avg rent by CMA (after {params.num_steps} years):")
    for cma in CMAS:
        start = CMA_BASE_RENTS[cma]
        end = results['avg_rent_by_cma'][cma][-1]
        pct = ((end - start) / start) * 100
        print(f"    {cma:<12}: ${start:,.0f} → ${end:,.0f}  (+{pct:.0f}%)")
    print("\n  Model smoke test passed.")
    return results

#Section to run emergence experiments
def run_emergence_experiments():
    """Runs all 3 emergence experiments and returns results."""
    section("STEP 2 — Emergence Experiments (E1, E2, E3)")
    from experiments_emergence import experiment_e1, experiment_e2, experiment_e3

    print("\n  Running E1: Rent Increase Rate vs. Displacement Emergence...")
    t0 = time.time()
    e1 = experiment_e1()
    print(f"    E1 completed in {time.time()-t0:.1f}s")

    print("\n  Running E2: Newcomer Bias vs. Income Gap Emergence...")
    t0 = time.time()
    e2 = experiment_e2()
    print(f"    E2 completed in {time.time()-t0:.1f}s")

    print("\n  Running E3: Rent-Income Gap vs. Affordability Collapse...")
    t0 = time.time()
    e3 = experiment_e3()
    print(f"    E3 completed in {time.time()-t0:.1f}s")

    return {'e1': e1, 'e2': e2, 'e3': e3}

# Section to run self-organization experiments
def run_so_experiments():
    """Runs all 3 self-organization experiments and returns results."""
    section("STEP 3 — Self-Organization Experiments (S1, S2, S3)")
    from experiments_so import experiment_s1, experiment_s2, experiment_s3

    print("\n  Running S1: Affordability Threshold vs. Household Self-Sorting...")
    t0 = time.time()
    s1 = experiment_s1()
    print(f"    S1 completed in {time.time()-t0:.1f}s")

    print("\n  Running S2: Employer Removal vs. Labour Network Reorganization...")
    t0 = time.time()
    s2 = experiment_s2()
    print(f"    S2 completed in {time.time()-t0:.1f}s")

    print("\n  Running S3: Rent Cap Policy Shock vs. Network Re-equilibration...")
    t0 = time.time()
    s3 = experiment_s3()
    print(f"    S3 completed in {time.time()-t0:.1f}s")

    return {'s1': s1, 's2': s2, 's3': s3}

# Section to print summary of results from emergence,  self-organization experiments and output files
def print_summary(emergence_results, so_results):
    print("  Output files saved to outputs/:")
    graphs = [
        'E1_rent_rate_displacement_emergence.png',
        'E2_newcomer_bias_emergence.png',
        'E3_affordability_collapse_emergence.png',
        'S1_affordability_threshold_self_sorting.png',
        'S2_employer_removal_SO.png',
        'S3_rent_cap_SO_reequilibration.png',
    ]
    for g in graphs:
        path = os.path.join(OUTPUT_DIR, g)
        exists = "Yes" if os.path.exists(path) else "MISSING"
        print(f"    {exists}  {g}")

# Main function to run the model test, emergence experiments, and self-organization experiments
def main():
    args = sys.argv[1:]

    if '--model' in args:
        run_model_test()
        return

    if '--emergence' in args:
        e_results = run_emergence_experiments()
        section("EMERGENCE EXPERIMENTS COMPLETE")
        print(f"  3 graphs saved to: {OUTPUT_DIR}")
        return

    if '--so' in args:
        so_results = run_so_experiments()
        section("SO EXPERIMENTS COMPLETE")
        print(f"  3 graphs saved to: {OUTPUT_DIR}")
        return

    total_start = time.time()

    model_results = run_model_test()
    emergence_results = run_emergence_experiments()
    so_results = run_so_experiments()

    print_summary(emergence_results, so_results)

    section("ALL EXPERIMENTS COMPLETE")
    total_time = time.time() - total_start
    print(f"\n  Total runtime   : {total_time:.1f} seconds")
    print(f"  Graphs saved to : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()