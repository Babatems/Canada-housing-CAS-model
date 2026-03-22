PROJECT OVERVIEW:
This project models Canada's housing affordability and rent crisis as a Complex
Adaptive System (CAS) using a network/graph approach in Python 3.11.14. Household,
neighbourhood, and employer agents interact on a weighted graph calibrated with
real Canadian open-source data. Six experiments demonstrate emergence (E1, E2,
E3) and self-organization (S1, S2, S3) by varying four core model parameters.

Datasets used:
  - Statistics Canada Table 46-10-0072-01 (Housing Acceptability by Immigrant
    Status) — https://www150.statcan.gc.ca/n1/tbl/csv/46100072-eng.zip
  - Statistics Canada Table 18-10-0205-01 (New Housing Price Index, Monthly)
    — https://www150.statcan.gc.ca/n1/tbl/csv/18100205-eng.zip
  - CMHC Rental Market Survey — Average Apartment Rents (Vacant & Occupied)
    — https://www.cmhc-schl.gc.ca/professionals/housing-markets-data-and-research/housing-data/data-tables/rental-market
  - CMHC Rental Market Survey — Urban Vacancy Rates
    — same URL as above
  - CMHC Rental Market Survey — Vacancy Rate by Rent Quartile
    — same URL as above

FOLDER STRUCTURE:
final_project/
  |
  |-- papers/
  |     Contains the 6 annotated PDF papers reviewed for this project.
  |
  |-- data/
  |     |-- housing-acceptable-data/
  |     |     statcan_housing_acceptable.csv
  |     |-- hpi-data/
  |     |     statcan_housing_price_index.csv
  |     |-- average-rents-vacant-occupied-units-2020-en.xlsx
  |     |-- urban-rental-market-survey-data-vacancy-rates-2023-en.xlsx
  |     |-- urban-rental-market-survey-data-vacancy-rates-rent-quartile-2023-en.xlsx
  |
  |-- src/
  |     data_loader.py              Loads and previews all 5 datasets
  |     model.py                    CAS network model + agent definitions
  |     experiments_emergence.py    3 emergence experiments (E1, E2, E3)
  |     experiments_so.py           3 self-organization experiments (S1, S2, S3)
  |     main.py                     Single entry point — runs everything
  |
  |-- outputs/
  |     E1_rent_rate_displacement_emergence.png
  |     E2_newcomer_bias_emergence.png
  |     E3_affordability_collapse_emergence.png
  |     S1_affordability_threshold_self_sorting.png
  |     S2_employer_removal_SO.png
  |     S3_rent_cap_SO_reequilibration.png
  |
  |-- README.txt                    This file you are currently in
  |-- requirements.txt              Conda environment dependencies

ENVIRONMENT SETUP:
This project requires Python 3.11.14. A conda environment is recommended.

Step 1 — Create and activate the conda environment:

    conda create -n final_project python=3.11.14
    conda activate final_project

Step 2 — Install all dependencies from requirements.txt:

    pip install -r requirements.txt

Step 3 — Verify Python version:

    python --version
    # Expected output: Python 3.11.14

HOW TO RUN:
All commands should be run from inside the src/ directory:

    cd src/

--- Option 1: Run ALL experiments (recommended for full demo) ---

    python main.py

    This runs the model smoke test, all 3 emergence experiments, and all 3
    self-organization experiments. All 6 output graphs are saved to outputs/.
    Total runtime: approximately 2-3 seconds.

--- Option 2: Run only the model smoke test ---

    python main.py --model

--- Option 3: Run only emergence experiments ---

    python main.py --emergence

--- Option 4: Run only self-organization experiments ---

    python main.py --so

--- Option 5: Inspect datasets only ---

    python data_loader.py

    Loads and previews all 5 datasets. Useful for verifying data files are
    correctly placed before running the model.

--- Option 6: Run individual experiment files ---

    python experiments_emergence.py
    python experiments_so.py

EXPECTED OUTPUT:
Running python main.py will produce the following terminal output sections:

  STEP 1 — Model Smoke Test
    Confirms network builds correctly (176 nodes, 300 edges).
    Prints step-by-step simulation table for 10 years.
    Prints final average rent by CMA.

  STEP 2 — Emergence Experiments (E1, E2, E3)
    E1: 5 rent rate scenarios — displacement rates from 11.33% to 100%.
    E2: 5 bias levels — newcomer displacement gap from -2.9pp to +36pp.
    E3: 5 rent-income gap scenarios — displacement from 12% to 100%.

  STEP 3 — Self-Organization Experiments (S1, S2, S3)
    S1: 5 affordability thresholds — SO concentration peaks at 30%.
    S2: 5 employer removal levels — newcomer displacement grows to 75.56%.
    S3: 5 rent cap timing scenarios — early cap reduces displacement by 75%.

  EXPERIMENT SUMMARY TABLE
    Prints a full pass/fail summary of all 6 experiments.

  6 PNG graphs saved to outputs/:
    E1_rent_rate_displacement_emergence.png
    E2_newcomer_bias_emergence.png
    E3_affordability_collapse_emergence.png
    S1_affordability_threshold_self_sorting.png
    S2_employer_removal_SO.png
    S3_rent_cap_SO_reequilibration.png

DATASET DOWNLOAD INSTRUCTIONS:
If the data/ folder is empty (e.g. after cloning from GitHub), download the
datasets manually as follows:

1. Statistics Canada Table 46-10-0072-01:
   - Go to: https://www150.statcan.gc.ca/n1/tbl/csv/46100072-eng.zip
   - Extract and rename the CSV to: statcan_housing_acceptable.csv
   - Place in: data/housing-acceptable-data/

2. Statistics Canada Table 18-10-0205-01:
   - Go to: https://www150.statcan.gc.ca/n1/tbl/csv/18100205-eng.zip
   - Extract and rename the CSV to: statcan_housing_price_index.csv
   - Place in: data/hpi-data/

3. CMHC Average Rents (Vacant & Occupied):
   - Go to: https://www.cmhc-schl.gc.ca/professionals/housing-markets-data-
     and-research/housing-data/data-tables/rental-market
   - Download: Average Apartment Rents (Vacant & Occupied)
   - Place in: data/

4. CMHC Vacancy Rates:
   - Same CMHC page as above
   - Download: Vacancy Rates (Urban, centres 10,000+)
   - Place in: data/

5. CMHC Vacancy Rate by Rent Quartile:
   - Same CMHC page as above
   - Download: Vacancy Rate by Rent Quartile
   - Place in: data/