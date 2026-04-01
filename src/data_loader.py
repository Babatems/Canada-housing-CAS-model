import os
import pandas as pd
import numpy as np

# Path to the data directory, relative to this script
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

#Renaming files to a dictionary for easy access and maintenance
FILES = {
    "statcan_housing": "housing-acceptable-data/statcan_housing_acceptable.csv",
    "statcan_price_index": "hpi-data/statcan_housing_price_index.csv",
    "cmhc_avg_rents": "average-rents-vacant-occupied-units-2020-en.xlsx",
    "cmhc_vacancy": "urban-rental-market-survey-data-vacancy-rates-2023-en.xlsx",
    "cmhc_vacancy_quartile": "urban-rental-market-survey-data-vacancy-rates-rent-quartile-2023-en.xlsx"
}

CMHC_RENT_COLUMNS = [
    'zone', 'year',
    'bachelor_vacant', 'bachelor_vacant_flag', 'bachelor_occupied', 'bachelor_occupied_flag', 'bachelor_sig', 'bedroom1_vacant', 'bedroom1_vacant_flag', 'bedroom1_occupied', 'bedroom1_occupied_flag', 'bedroom1_sig', 'bedroom2_vacant', 'bedroom2_vacant_flag', 'bedroom2_occupied', 'bedroom2_occupied_flag', 'bedroom2_sig', 'bedroom3_vacant', 'bedroom3_vacant_flag', 'bedroom3_occupied', 'bedroom3_occupied_flag', 'bedroom3_sig', 'total_vacant', 'total_vacant_flag', 'total_occupied', 'total_occupied_flag', 'total_sig'
]

#header function to find the header row in an Excel file based on a keyword
def find_header_row(path, keyword, sheet=0, max_rows=20):
    raw = pd.read_excel(path, sheet_name=sheet, header=None, nrows=max_rows, engine='openpyxl')
    for i, row in raw.iterrows():
        if row.astype(str).str.contains(keyword, case=False, na=False).any():
            return i
    return 0

#A loader function to read Excel files with dynamic header row detection, and handle errors.
def load_xlsx_smart(key, header_keyword, sheet=0):
    path = os.path.join(DATA_DIR, FILES[key])
    try:
        header_row = find_header_row(path, header_keyword, sheet=sheet)
        df = pd.read_excel(path, sheet_name=sheet, skiprows=header_row, engine='openpyxl')
        print(f"\n Loaded [{key}] — header at row {header_row} — "
              f"{df.shape[0]} rows x {df.shape[1]} cols")
        return df
    except FileNotFoundError:
        print(f"\n File not found for [{key}]: {path}")
        return None
    except Exception as e:
        print(f"\n Error loading [{key}]: {e}")
        return None

#A loader function to read CSV files with error handling and encoding fallbacks.
def load_csv(key):
    path = os.path.join(DATA_DIR, FILES[key])
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
        print(f"\n Loaded [{key}] — {df.shape[0]} rows x {df.shape[1]} cols")
        return df
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin1')
        print(f"\n Loaded [{key}] — {df.shape[0]} rows x {df.shape[1]} cols")
        return df
    except FileNotFoundError:
        print(f"\n File not found for [{key}]: {path}")
        return None
    except Exception as e:
        print(f"\n Error loading [{key}]: {e}")
        return None

#A fucntion to help clean up column names. Strips whitespace, converts to lowercase, replaces spaces and newlines with underscores, and removes special characters.
def clean_columns(df):
    df.columns = (df.columns
                  .astype(str)
                  .str.strip()
                  .str.lower()
                  .str.replace(r'[\s\n]+', '_', regex=True)
                  .str.replace(r'[^\w_]', '', regex=True)
    )
    return df

#A function to convert columns to numeric format 
def to_numeric_safe(series):
    return pd.to_numeric(
        series.astype(str).str.replace('--', '', regex=False).str.strip(),
        errors='coerce'
    )

#Function to load Statcan housing dataset
def load_statcan_housing():
    """
    Statistics Canada — Persons in acceptable/unacceptable housing.
    """
    df = load_csv("statcan_housing")
    if df is None:
        return None
    df = clean_columns(df)
    df.rename(columns={df.columns[0]: 'ref_date'}, inplace=True)
    df = df.dropna(subset=['value'])
    if 'statistics' in df.columns:
        df = df[df['statistics'].str.strip() == 'Percentage of persons'].copy()
    return df.reset_index(drop=True)

#Funcrion to load Statcan housing price index dataset
def load_statcan_price_index():
    """
    Statistics Canada — New Housing Price Index, monthly by CMA.
    """
    df = load_csv("statcan_price_index")
    if df is None:
        return None
    df = clean_columns(df)
    df.rename(columns={df.columns[0]: 'ref_date'}, inplace=True)
    df['ref_date'] = pd.to_datetime(df['ref_date'], errors='coerce')
    df = df.dropna(subset=['value'])
    if 'new_housing_price_indexes' in df.columns:
        df = df[df['new_housing_price_indexes'].str.strip() == 'Total (house and land)'].copy()
    return df.reset_index(drop=True)

#Function to load CMHC average rent dataset.
def load_cmhc_avg_rents():
    """
    CMHC — Average Apartment Rents (Vacant & Occupied).
    """
    path = os.path.join(DATA_DIR, FILES["cmhc_avg_rents"])
    try:
        header_row = find_header_row(path, "bachelor", sheet=0)
        df = pd.read_excel(path, sheet_name=0, skiprows=header_row + 1, header=None, engine='openpyxl')
        if df.shape[1] == len(CMHC_RENT_COLUMNS):
            df.columns = CMHC_RENT_COLUMNS
        else:
            cols = CMHC_RENT_COLUMNS[:df.shape[1]]
            cols += [f'extra_{i}' for i in range(df.shape[1] - len(cols))]
            df.columns = cols

        print(f"\n Loaded [cmhc_avg_rents] — {df.shape[0]} rows x {df.shape[1]} cols")

        df = df[df['zone'].astype(str).str.lower() != 'zone'].copy()
        df = df.dropna(subset=['zone']).reset_index(drop=True)

        df = df[df['zone'].astype(str).str.len() < 60].copy()

        rent_cols = ['bachelor_occupied', 'bedroom1_occupied',
                     'bedroom2_occupied', 'bedroom3_occupied', 'total_occupied']
        for col in rent_cols:
            if col in df.columns:
                df[col] = to_numeric_safe(df[col])

        return df.reset_index(drop=True)

    except FileNotFoundError:
        print(f"\n File not found for [cmhc_avg_rents]: {path}")
        return None
    except Exception as e:
        print(f"\n Error loading [cmhc_avg_rents]: {e}")
        return None

#FUnction to load CMHC vacancy dataset
def load_cmhc_vacancy():
    """
    CMHC — Urban Rental Market Vacancy Rates.
    """
    df = load_xlsx_smart("cmhc_vacancy", header_keyword="Province")
    if df is None:
        return None
    df = clean_columns(df)
    df = df.dropna(how='all').reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.startswith('unnamed')]
    first_col = df.columns[0]
    df = df[df[first_col].notna()].reset_index(drop=True)

    for col in ['bachelor', '1_bedroom', '2_bedroom', '3_bedroom_', 'total']:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])
    return df

#Function to load CMHC vacancy by rent quartile dataset
def load_cmhc_vacancy_quartile():
    """
    CMHC — Vacancy Rate by Rent Quartile.
    """
    df = load_xlsx_smart("cmhc_vacancy_quartile", header_keyword="Province")
    if df is None:
        return None
    df = clean_columns(df)
    df = df.dropna(how='all').reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.startswith('unnamed')]
    first_col = df.columns[0]
    df = df[df[first_col].notna()].reset_index(drop=True)
    for col in ['1st', '2nd', '3rd', '4th']:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])
    return df

#Function to print a preview of each data set including their attributes.
def preview(name, df):
    if df is None:
        print(f"\n  [{name}] failed to load.")
        return
    print(f"\n{'='*60}")
    print(f"  DATASET : {name}")
    print(f"{'='*60}")
    print(f"  Shape   : {df.shape}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Sample  :")
    print(df.head(3).to_string(index=False))
    print(f"  Nulls   : {df.isnull().sum().sum()} total missing values")


# MAIN loader function to load all datasets and print previews

def load_all():
    print("\n" + "="*60)
    print("  LOADING ALL DATASETS")
    print("="*60)

    datasets = {
        "StatCan Housing Acceptability" : load_statcan_housing(),
        "StatCan Housing Price Index" : load_statcan_price_index(),
        "CMHC Average Rents" : load_cmhc_avg_rents(),
        "CMHC Vacancy Rates" : load_cmhc_vacancy(),
        "CMHC Vacancy by Rent Quartile":  load_cmhc_vacancy_quartile(),
    }

    print("\n\n" + "="*60)
    print("  DATASET PREVIEWS")
    print("="*60)
    for name, df in datasets.items():
        preview(name, df)

    return datasets


if __name__ == "__main__":
    datasets = load_all()
    print("\n\n All datasets loaded and cleaned.")