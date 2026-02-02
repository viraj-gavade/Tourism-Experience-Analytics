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
# Merge with cities - this will create CountryId_x (from users) and CountryId_y (from cities)
df = df.merge(cities, on="CityId", how="left", suffixes=("", "_city"))

# Keep CountryId from users (already has the right one), drop the one from cities
if "CountryId_city" in df.columns:
    df = df.drop(columns=["CountryId_city"])

# Now merge with countries on CountryId (from users)
df = df.merge(countries, on="CountryId", how="left", suffixes=("", "_country"))

# Countries has RegionId and ContinentId - keep the ones from users
if "RegionId_country" in df.columns:
    df = df.drop(columns=["RegionId_country"])
if "ContinentId_country" in df.columns:
    df = df.drop(columns=["ContinentId_country"])

# Merge with regions
df = df.merge(regions, on="RegionId", how="left", suffixes=("", "_region"))

# Regions has ContinentId - keep the one from users
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
# Rename VisitMode to VisitModeId in transactions data for proper merge
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
# 8. Drop ID Columns
# ---------------------------
drop_cols = [
    "TransactionId","UserId",
    "CityId","CountryId","RegionId","ContinentId",
    "AttractionId","AttractionTypeId",
    "VisitModeId"
]

df.drop(columns=drop_cols, inplace=True, errors="ignore")

# ---------------------------
# 9. Save Final ML Dataset
# ---------------------------
df.to_csv("data/processed/master_ml_dataset.csv", index=False)

print("\nMaster ML Dataset Saved as master_ml_dataset.csv")
print("Script Completed Successfully")
