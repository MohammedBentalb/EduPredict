# %%
import pandas as pd
import seaborn as sns
from data_profiling import ProfileReport
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent # makhdamach f notbook

# %%
def load_data():
    return pd.read_csv(ROOT / "Data/Raw/dataset.csv")
# %%
def fill_with_mode(df: pd.DataFrame, column: str):
    df = df.copy()
    parent_education_mode = df[column].mode()[0]
    df.loc[df[column].isna(), column] = parent_education_mode
    return df
# %%
def find_outliers(df: pd.DataFrame, col):
  Q1, Q3 = df[col].quantile([0.25, 0.75])
  IQR = Q3 - Q1
  MIN_VALUE = Q1 - 1.5 * IQR
  MAX_VALUE = Q3 + 1.5 * IQR
  mask = (df[col] < MIN_VALUE) | (df[col] > MAX_VALUE)
  outliers = df[mask]
  return len(outliers), outliers, IQR, Q1, Q3, mask
# %%
def nominal_encoding(df: pd.DataFrame, cols):
  t = pd.get_dummies(df, columns=cols ,drop_first=True, dtype=int)
  return t

# %%
def ordinal_encoding(df: pd.DataFrame, mappings: dict):
    df = df.copy()
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)
    return df
# %%
def drop_duplicates(df: pd.DataFrame):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df
# %%
def process_data():
    ordinal_mappings = {
        "Parental_Involvement": {"Low": 0, "Medium": 1, "High": 2},
        "Access_to_Resources": {"Low": 0, "Medium": 1, "High": 2},
        "Motivation_Level": {"Low": 0, "Medium": 1, "High": 2},
        "Family_Income": {"Low": 0, "Medium": 1, "High": 2},
        "Teacher_Quality": {"Low": 0, "Medium": 1, "High": 2},
        "Parental_Education_Level": {"High School": 0, "College": 1, "Postgraduate": 2},
        "Distance_from_Home": {"Near": 0, "Moderate": 1, "Far": 2},
    }

    nominal_cols = ["Gender", "School_Type", "Extracurricular_Activities",
                    "Internet_Access", "Learning_Disabilities", "Peer_Influence"]

    df = load_data()
    for col in ["Parental_Education_Level", "Distance_from_Home", "Teacher_Quality"]:
        df = fill_with_mode(df,col)

    df = nominal_encoding(df, nominal_cols)
    df = ordinal_encoding(df, ordinal_mappings)
    df = drop_duplicates(df)
    return df