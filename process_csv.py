import pandas as pd
import json

# Load the CSV
df = pd.read_csv('india_crop_yield_with_final.csv')

# Get unique combinations of State, District, Soil_Type
unique_combinations = df[['State', 'District', 'Soil_Type']].drop_duplicates()

# Group by State and District, collect unique soil types
state_district_soil = {}
for state in unique_combinations['State'].unique():
    state_data = {}
    state_df = unique_combinations[unique_combinations['State'] == state]
    for district in state_df['District'].unique():
        soils = state_df[state_df['District'] == district]['Soil_Type'].unique().tolist()
        state_data[district] = soils
    state_district_soil[state] = state_data

# Save to JSON
with open('state_district_soil.json', 'w') as f:
    json.dump(state_district_soil, f, indent=4)

print("JSON created: state_district_soil.json")
