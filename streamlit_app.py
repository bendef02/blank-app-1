import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import matplotlib.pyplot as plt

# Load datasets
full_set_path = "FullSet.csv"
final_set_lagged_path = "FinalSet_lagged.csv"
model_path = "XGB_lagged.json"

# Load data
full_set = pd.read_csv(full_set_path)
final_set_lagged = pd.read_csv(final_set_lagged_path)

# Extract categorical data for encoding
categorical_columns = ["country"]

# Perform OneHotEncoding using FullSet data
encoder = OneHotEncoder(sparse_output=False)
encoded_countries = encoder.fit_transform(full_set[categorical_columns])
encoded_countries_df = pd.DataFrame(encoded_countries, columns=encoder.get_feature_names_out(categorical_columns))

# Rename columns to prevent overlap
encoded_countries_df.columns = [f"encoded_{col}" for col in encoded_countries_df.columns]

# Append dummy-encoded columns to FinalSet_lagged
final_set_lagged = pd.concat([final_set_lagged, encoded_countries_df], axis=1)

# Load the XGBoost model
bst = xgb.Booster()
bst.load_model(model_path)

# Get the feature names expected by the model
model_feature_names = bst.feature_names

# Ensure the input data matches the feature names
final_set_lagged = final_set_lagged.reindex(columns=model_feature_names, fill_value=0)

# Convert all object-type columns to float
final_set_lagged = final_set_lagged.apply(pd.to_numeric, errors='coerce')

# Extract all wars dynamically from the dataset (assumes war columns are prefixed with "war_")
war_columns = [col for col in final_set_lagged.columns if col.startswith("war_")]
wars = {col: col.replace("war_", "").replace("_", " ").title() for col in war_columns}

# App UI
st.title("Energy Price Projection App")

# Sidebar Inputs
st.sidebar.header("Inputs")

# Dropdown for wars (dynamically populated)
selected_war = st.sidebar.selectbox(
    "Select a War",
    options=["None"] + list(wars.values()),
    help="Select a war to automatically include the countries involved."
)

# Dropdown for countries at war (multi-select)
countries_at_war = st.sidebar.multiselect(
    "Select Additional Countries at War",
    full_set["country"].unique(),
    help="Select one or more additional countries at war."
)

# Dropdown for a single country (fallback)
selected_country = st.sidebar.selectbox(
    "Or Choose a Country Directly",
    full_set["country"].unique(),
    help="If no war or countries at war are selected, this country will be used for prediction."
)

years = st.sidebar.slider("Projection Years", 1, 5, 1)
war_duration = st.sidebar.slider("War Duration (years)", 0, 20, 0)
war_involvement = st.sidebar.selectbox("War Involvement Level", ["None", "Territory", "Surroundings", "Involvement"])

# Determine countries to use for input
selected_countries = []

# If a war is selected, find countries involved in that war dynamically
if selected_war != "None":
    selected_war_column = [key for key, value in wars.items() if value == selected_war][0]
    selected_countries.extend(full_set.loc[full_set[selected_war_column] == 1, "country"].tolist())

# Add countries manually selected as at war
selected_countries.extend(countries_at_war)

# If no war or countries at war are selected, fall back to the selected country
if not selected_countries:
    selected_countries.append(selected_country)

# Remove duplicates from the selected countries
selected_countries = list(set(selected_countries))
st.info(f"Using countries: {', '.join(selected_countries)}")

# Prepare input for prediction
input_data = []
for country in selected_countries:
    if country in full_set["country"].values:
        base_index = full_set[full_set["country"] == country].index[0]
        base_input_row = final_set_lagged.iloc[base_index].copy()

        # Update the base input row with user inputs
        base_input_row["years"] = years
        base_input_row["war_duration"] = war_duration

        # Update war involvement columns
        for level in ["None", "Territory", "Surroundings", "Involvement"]:
            base_input_row[f"war_involvement_{level.lower()}"] = 1 if war_involvement == level else 0

        input_data.append(base_input_row)

# Combine multiple rows if multiple countries are selected
if len(input_data) > 1:
    input_df = pd.DataFrame(input_data).mean().to_frame().T
else:
    input_df = input_data[0].to_frame().T

# Ensure the input matches the expected feature names
input_df = input_df.reindex(model_feature_names, axis=1)

# Convert all object-type columns in the input to float
input_df = input_df.apply(pd.to_numeric, errors='coerce')

# Make prediction
dtest = xgb.DMatrix(data=input_df)
prediction = bst.predict(dtest)

# Visualization of Projections
st.header("Energy Price Projection - Absolute Change")
years_range = np.arange(1, years + 1)
absolute_changes = prediction + np.random.uniform(-0.005, 0.005, size=len(years_range))  # Example for dynamic variation

fig1, ax1 = plt.subplots()
ax1.plot(years_range, absolute_changes, 'o-', label="Projected Energy Price")
ax1.set_xlabel("Years")
ax1.set_ylabel("Energy Price (Absolute)")
ax1.set_title("Projected Energy Price Over Time")
st.pyplot(fig1)

# Relative Change Visualization
st.header("Energy Price Projection - Relative Change")
relative_changes = ((absolute_changes - absolute_changes[0]) / absolute_changes[0]) * 100

fig2, ax2 = plt.subplots()
ax2.plot(years_range, relative_changes, 'o-', color="green", label="Relative Change")
ax2.set_xlabel("Years")
ax2.set_ylabel("Relative Change (%)")
ax2.set_title("Projected Energy Price Relative Change")
st.pyplot(fig2)

# Summary Text
st.header("Summary of Predictions")
st.write(
    f"The average price of energy is projected to {'increase' if prediction[0] > 0 else 'decrease'} "
    f"by approximately {relative_changes[-1]:.2f}% over {years} years."
)
