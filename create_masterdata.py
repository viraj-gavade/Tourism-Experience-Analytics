import pandas as pd
import os

# Create processed directory if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

# ---------------------------
# 1. Load Raw Data
# ---------------------------
transactions = pd.read_excel("data/raw/Transaction.xlsx")
users = pd.read_excel("data/raw/User.xlsx")
cities = pd.read_excel("data/raw/City.xlsx")
countries = pd.read_excel("data/raw/Country.xlsx")
regions = pd.read_excel("data/raw/Region.xlsx")
continents = pd.read_excel("data/raw/Continent.xlsx")
items = pd.read_excel("data/raw/Item.xlsx")
types = pd.read_excel("data/raw/Type.xlsx")
visitmode = pd.read_excel("data/raw/Mode.xlsx")

print("Datasets Loaded Successfully")

# ---------------------------
# 2. Create Recommender Dataset
# ---------------------------
recommender_df = transactions[["UserId","AttractionId","Rating"]]
recommender_df.to_csv("data/processed/recommender_dataset.csv", index=False)

print("Recommender Dataset Saved")

# ---------------------------
# 3. Merge Transaction + User
# ---------------------------
df = transactions.merge(users, on="UserId", how="left")

# ---------------------------
# 4. Merge Location Hierarchy
# ---------------------------
# Merge with cities
df = df.merge(cities, on="CityId", how="left", suffixes=("", "_city"))

# Keep CountryId from users, drop the one from cities
if "CountryId_city" in df.columns:
    df = df.drop(columns=["CountryId_city"])

# Merge with countries
df = df.merge(countries, on="CountryId", how="left", suffixes=("", "_country"))

# Keep RegionId and ContinentId from users
if "RegionId_country" in df.columns:
    df = df.drop(columns=["RegionId_country"])
if "ContinentId_country" in df.columns:
    df = df.drop(columns=["ContinentId_country"])

# Merge with regions
df = df.merge(regions, on="RegionId", how="left", suffixes=("", "_region"))

# Keep ContinentId from users
if "ContinentId_region" in df.columns:
    df = df.drop(columns=["ContinentId_region"])

# Merge with continents
df = df.merge(continents, on="ContinentId", how="left")

# ---------------------------
# 5. Merge Attraction Info
# ---------------------------
df = df.merge(items, on="AttractionId", how="left")
df = df.merge(types, on="AttractionTypeId", how="left")

# ---------------------------
# 6. Merge Visit Mode Labels
# ---------------------------
df = df.rename(columns={"VisitMode": "VisitModeId"})
df = df.merge(visitmode, on="VisitModeId", how="left")

# ---------------------------
# 7. Validation Checks
# ---------------------------
print("Rows in Transaction:", transactions.shape[0])
print("Rows After Merge:", df.shape[0])

print("\nMissing Values Check:")
print(df.isnull().sum())

# ---------------------------
# 8. Save FULL Dataset (WITH IDs) - FOR FEATURE ENGINEERING
# ---------------------------
# Save this one FIRST - it has all the ID columns we need!
df.to_csv("data/processed/master_dataset_with_ids.csv", index=False)
print("\n✅ Master Dataset WITH IDs Saved (for feature engineering)")

# ---------------------------
# 9. Drop ID Columns for Basic ML Dataset
# ---------------------------
drop_cols = [
    "TransactionId","UserId",
    "CityId","CountryId","RegionId","ContinentId",
    "AttractionId","AttractionTypeId",
    "VisitModeId"
]

df_no_ids = df.drop(columns=drop_cols, errors="ignore")

# ---------------------------
# 10. Save ML Dataset Without IDs (legacy version)
# ---------------------------
df_no_ids.to_csv("data/processed/master_ml_dataset.csv", index=False)

print("✅ Master ML Dataset (without IDs) Saved as master_ml_dataset.csv")
print("\nScript Completed Successfully")
print(f"\nTwo files created:")
print(f"  1. master_dataset_with_ids.csv ({df.shape[0]} rows, {df.shape[1]} cols) - USE THIS ONE")
print(f"  2. master_ml_dataset.csv ({df_no_ids.shape[0]} rows, {df_no_ids.shape[1]} cols) - Legacy")