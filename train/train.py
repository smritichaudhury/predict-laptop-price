import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

import warnings
warnings.filterwarnings('ignore')

# Print library versions
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Seaborn version: {sns.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")


"""

READING THE FILE

"""

file_path = 'train/laptop.csv'
raw_data = pd.read_csv(file_path)

# Drop unnamed 0 column
raw_data.drop('Unnamed: 0', axis=1, inplace=True)

"""

PRE PROCESSING THE FILE

"""

# Upper case all the column names to lower case
raw_data.columns = raw_data.columns.str.lower()

print(raw_data['processor_brand'].value_counts())

brands_to_exclude = ['Microsoft', 'Qualcomm', 'Intel Core', 'AMD', 'Apple', 'MediaTek', 'Intel']
filtered_data = raw_data[~raw_data['processor_brand'].isin(brands_to_exclude)]
filtered_data

print(filtered_data['processor_brand'].value_counts())
print(len(filtered_data))

# Remove the instances
raw_data = raw_data[~raw_data.index.isin(filtered_data.index)]
print(len(raw_data))
raw_data.head(2)

# RAM COLUMN
print(raw_data['ram'].value_counts())

# Function to extract numeric part from RAM string
def extract_numeric_ram(ram_str):
    numeric_part = ''.join(filter(str.isdigit, ram_str))
    if numeric_part:
        return int(numeric_part)
    else:
        return np.nan  # Handle cases with no numeric value


# Apply the function to the 'RAM' column
raw_data['ram'] = raw_data['ram'].apply(extract_numeric_ram)

# Convert the 'RAM' column to integer type
raw_data['ram'] = pd.to_numeric(raw_data['ram'], errors='coerce').astype('Int64') # Use 'Int64' for nullable integer

# Now you can check the updated 'RAM' column
print(raw_data['ram'].value_counts())

# Confirming the changes
raw_data.info()

# RAM_TYPE Imputation

# Strip whitespace from 'RAM_TYPE' column
raw_data['ram_type'] = raw_data['ram_type'].str.strip()

# Value counts
raw_data['ram_type'].value_counts()

ram_mapping = {
    91: 'DDR4',
    96: 'LPDDR5',
    222: 'DDR4',
    284: 'DDR4',
    358: 'DDR4',
    371: 'Unified',
    567: 'LPDDR5',
    568: 'LPDDR5',
    570: 'LPDDR5',
    572: 'LPDDR5',
    573: 'LPDDR5',
    576: 'LPDDR5',
    588: 'LPDDR5',
    591: 'LPDDR5',
    639: 'Unified',
    657: 'LPDDR5',
    658: 'LPDDR5',
    659: 'LPDDR5',
    660: 'LPDDR5',
    661: 'LPDDR5',
    699: 'Unified',
    950: 'Unified',
    1202: 'DDR4',
    1226: 'LPDDR5',
    1582: 'DDR4',
    1972: 'Unified',
    1978: 'Unified',
    1980: 'Unified',
    1981: 'Unified',
    1982: 'Unified',
    1983: 'Unified',
    2204: 'Unified',
    2233: 'Unified',
    2247: 'Unified',
    2292: 'LPDDR5',
    2293: 'LPDDR5',
    2320: 'Unified',
    3070: 'Unified',
    3316: 'DDR4'
}


for index, row in raw_data.iterrows():
    if row['ram_type'] in ram_mapping.values():
        continue  # Skip if already present in ram_mapping.values

    if index in ram_mapping:
        raw_data.loc[index, 'ram_type'] = ram_mapping[index]
        
# Remove "ram" from the 'ram_type' column
raw_data['ram_type'] = raw_data['ram_type'].str.replace('ram', '', regex=False)

# Display value counts to confirm changes
raw_data['ram_type'] = raw_data['ram_type'].str.replace(r'\s*RAM$', '', regex=True)

# Display value counts to confirm changes
print(raw_data['ram_type'].value_counts())


# Identify rows where 'Display' column contains "OLED Display With Touchscreen"
rows_to_drop = raw_data[raw_data['display'] == 'OLED Display With Touchscreen'].index

# Drop those rows from the DataFrame
raw_data.drop(rows_to_drop, inplace=True)

# Now, verify the changes
raw_data['display'].value_counts()

# Strip all the white spaces in DISPLAY column and then convert it into float type first
raw_data['display'] = raw_data['display'].str.strip()
raw_data['display'] = raw_data['display'].astype('float')

# RAM EXPANDABLE COLUMN
raw_data['ram_expandable'].value_counts()

# Replace 'NOT EXPANDABLE' with '0' and remove 'GB' and 'EXPANDABLE'
raw_data['ram_expandable'] = raw_data['ram_expandable'].astype(str).str.replace('NOT EXPANDABLE', '0', regex=False)
raw_data['ram_expandable'] = raw_data['ram_expandable'].str.replace('GB EXPANDABLE', '', regex=False)
# raw_data['RAM_EXPANDABLE'] = raw_data['RAM_EXPANDABLE'].str.replace('ALL', '', regex=False)

# Remove any non-numeric characters and convert to integers
raw_data['ram_expandable'] = raw_data['ram_expandable'].str.extract(r'(\d+)').fillna(0).astype(int)

# GPU_BRAND

# Convert 'GPU_BRAND' column to lowercase
raw_data['gpu_brand'] = raw_data['gpu_brand'].str.lower()

# Replace 'nividia' with 'nvidia' in the 'GPU_BRAND' column
raw_data['gpu_brand'] = raw_data['gpu_brand'].replace('nividia', 'nvidia')

# HDD
raw_data['hdd'].value_counts()

# Replace 'No HDD' with '0'
raw_data['hdd'] = raw_data['hdd'].astype(str).str.replace('No HDD', '0', regex=False)

# Remove 'GB HDD STORAGE' and convert to integer
raw_data['hdd'] = raw_data['hdd'].str.replace('GB HDD STORAGE', '', regex=False)

# Remove 'GB HDD STORAGE' and convert to integer
raw_data['hdd'] = raw_data['hdd'].str.lower().str.strip()
raw_data['hdd'] = raw_data['hdd'].str.replace('gb hdd storage', '', regex=False)

# Convert HDD column to integer type
raw_data['hdd'] = raw_data['hdd'].astype('int')

# SSD COLUMN
raw_data['ssd'] = raw_data['ssd'].str.lower().str.strip()

# Replace 'No SSD' with '0'
raw_data['ssd'] = raw_data['ssd'].astype(str).str.replace('no ssd', '0', regex=False)

# The \s* will match zero or more spaces before 'gb ssd storage'
raw_data['ssd'] = raw_data['ssd'].str.replace(r'\s*gb ssd storage', '', regex=True)

# Extract numeric values using regex and handle non-numeric values
raw_data['ssd'] = raw_data['ssd'].str.extract(r'(\d+)').fillna(0).astype(int)

# GHz Column
raw_data['ghz'].value_counts()

# Remove instances where 'GHZ' is 0
raw_data = raw_data[raw_data['ghz'] != '0']
raw_data['ghz'] = raw_data['ghz'].str.lower().str.strip()

# remove ghz processor
raw_data['ghz'] = raw_data['ghz'].str.replace('ghz processor', '', regex=False)

# Convert the GHZ column to integer type
raw_data['ghz'] = raw_data['ghz'].str.strip().astype('float')

"""

FEATURE SET SELECTION

"""

df = raw_data.copy()
df.head(2)


def lowercase_dataframe(df):


    df_lower = df.copy()  # Create a copy to avoid modifying the original DataFrame

    # Lowercase column names
    df_lower.columns = df_lower.columns.str.lower()

    for col in df_lower.columns:
        if df_lower[col].dtype == 'object':  # Check if the column is of 'object' dtype (string)
            try:
                df_lower[col] = df_lower[col].str.lower()
            except AttributeError:
                print(f"Column '{col}' cannot be converted to lowercase. It may contain non-string values.")
        elif pd.api.types.is_categorical_dtype(df_lower[col]):
            try:
                df_lower[col] = df_lower[col].astype(str).str.lower().astype('category')
            except AttributeError:
                print(f"Column '{col}' cannot be converted to lowercase. It may contain non-string values.")

    return df_lower

df = lowercase_dataframe(df)
df.head()


# Defining x and y
x = df.iloc[:, [4,5,6,7,8,9,10,12,13,14]]
y = df.iloc[:, 2]

categorical_cols = ['processor_brand', 'ram_type', 'display_type', 'gpu_brand']

# Perform dummy coding (one-hot encoding)
x = pd.get_dummies(x, columns=categorical_cols, drop_first=True)

# replace true with 1 and false with 0
x.replace({True: 1, False: 0}, inplace=True)

# Plotting the correlation matrix
plt.figure(figsize=(20, 10))
sns.heatmap(x.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # 80% train, 20% test

print(f"Training Features Shape: {x_train.shape}")
print(f"Test Features Shape: {x_test.shape}")
print(f"Training Labels Shape: {y_train.shape}")
print(f"Test Labels Shape: {y_test.shape}")


# Initialize the scaler
scaler = StandardScaler()

# Scale the training data (fit and transform)
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

"""

Lasso Regression Model using Grid Search CV

"""

# Define the range of alpha values for hyperparameter tuning
alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 0.9, 100.0]

# Initialize the Lasso model and GridSearchCV
lasso = Lasso(random_state=42, max_iter=10000)
param_grid = {'alpha': alpha_values}
grid_search = GridSearchCV(lasso, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)

# Perform grid search on the scaled training data
grid_search.fit(x_train_scaled, y_train)

# Get the best alpha value from the grid search
best_alpha = grid_search.best_params_['alpha']
print(f"Best alpha: {best_alpha}")

# Train the Lasso model using the best alpha value
best_lasso = Lasso(alpha=best_alpha, random_state=42, max_iter=10000)
best_lasso.fit(x_train_scaled, y_train)

# Predict on the test set
y_pred = best_lasso.predict(x_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
lasso_r2 = r2_score(y_test, y_pred)

print("\nLasso Regression Metrics with Best Alpha:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {lasso_r2}")


"""

Decision Tree Model

"""

# Initialize and train the Decision Tree model
dt_model = DecisionTreeRegressor(max_depth=4)
dt_model.fit(x_train_scaled, y_train)

# Make predictions
y_pred = dt_model.predict(x_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
dt_r2 = r2_score(y_test, y_pred)

print("Decision Tree Regression Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {dt_r2}")

"""
Random Forest Model

"""

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(x_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(x_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rf_r2 = r2_score(y_test, y_pred)

print("Random Forest Regression Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {rf_r2}")

"""

Random Forest Regressor Model using Randomized Search CV

"""

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300, 500],  # Increase the range for number of trees
    'max_depth': [5, 10, 20, 30, None],  # Add deeper trees and no limit
    'min_samples_split': [2, 5, 10],  # Minimum samples for splitting
    'min_samples_leaf': [1, 2, 5],  # Minimum samples for leaf nodes
    'max_features': ['sqrt', 'log2', None]  # Number of features to consider for best split
}

# Create a RandomForestRegressor model
rf_model = RandomForestRegressor(random_state=42)

# Use RandomizedSearchCV for broader exploration
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid,
                                   n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                   verbose=2, n_jobs=-1, random_state=42)

# Fit the model
random_search.fit(x_train_scaled, y_train)

# Get the best hyperparameters and model
best_params = random_search.best_params_
best_rf_model = random_search.best_estimator_

print("Best Hyperparameters:", best_params)

# Train the model with the best hyperparameters on the entire training data
best_rf_model.fit(x_train_scaled, y_train)

# Make predictions on the test set
y_pred = best_rf_model.predict(x_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
best_rf_r2 = r2_score(y_test, y_pred)

print("\nRandom Forest Regression Metrics (Best Hyperparameters):")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {best_rf_r2}")

"""

Refined Random Forest Model

"""

# Adjusted parameter grid for finer tuning
param_grid = {
    'n_estimators': [600, 800, 1000],  # Increased range
    'max_depth': [20, 30, 50, None],  # Deeper trees
    'min_samples_split': [5, 8, 10],  # Narrower range
    'min_samples_leaf': [1, 2, 3],  # Smaller values
    'max_features': ['sqrt', 'log2']  # Fixed options
}

# Randomized search for broader exploration
random_search = RandomizedSearchCV(estimator=best_rf_model, param_distributions=param_grid,
                                   n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                   verbose=2, n_jobs=-1, random_state=42)

random_search.fit(x_train_scaled, y_train)

# Evaluate the refined model
model = random_search.best_estimator_
y_pred = model.predict(x_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
best_rf_tuned_r2 = r2_score(y_test, y_pred)

print("\nRefined Random Forest Regression Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {best_rf_tuned_r2}")

# Comparison of models
r2_scores = {
    'Lasso': lasso_r2,
    'Decision Tree': dt_r2,
    'Random Forest': rf_r2,
    'Random Forest (RandomizedSearchCV)': best_rf_r2,
    'Random Forest (Tuned)': best_rf_tuned_r2
}

r2_df = pd.DataFrame.from_dict(r2_scores, orient='index', columns=['R2 Score'])

# Convert R2 scores to percentages
r2_df['R2 Score (%)'] = (r2_df['R2 Score'] * 100).round(2)

# Display the updated DataFrame
r2_df

# Get feature names AFTER one-hot encoding
feature_names = x_train.columns.astype(str).str.lower().tolist()

"""

Dump the model, scale and feature names to pickle file.

"""

# Save the model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL) # Use HIGHEST_PROTOCOL

# Save the scaler
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

# Save feature names
with open('model/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f, protocol=pickle.HIGHEST_PROTOCOL)

"""

Load the pickle files and predict the price

"""

def one_hot_encode_fixed_categories(input_data):
    """One-hot encodes categorical features with fixed categories (case-insensitive)."""
    all_categories = {
        'processor_brand': ['intel', 'amd', 'apple', 'mediatek', 'qualcomm', 'microsoft'],
        'ram_type': ['ddr4', 'ddr5', 'lpddr5', 'lpddr4x', 'ddr3', 'lpddr5x', 'lpddr3', 'lpddr4', 'ddr2'],
        'display_type': ['lcd', 'led'],
        'gpu_brand': ['intel', 'nvidia', 'amd', 'apple', 'mediatek', 'qualcomm', 'arm', 'microsoft', 'ati']
    }

    input_data = {k: str(v).lower() if isinstance(v, str) else v for k, v in input_data.items()}
    input_df = pd.DataFrame([input_data])

    for col, categories in all_categories.items():
        if col in input_df.columns:
            input_value = input_df[col].iloc[0]
            for category in categories:
                new_col_name = f"{col}_{category}"
                input_df[new_col_name] = 0
                if input_value == category:
                    input_df[new_col_name] = 1
            input_df = input_df.drop(col, axis=1)
        else:
            for category in categories:
                new_col_name = f"{col}_{category}"
                input_df[new_col_name] = 0
    return input_df

# Load model, scaler, and feature names
try:
    with open('model/model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    with open('model/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    # Exit if files are not found
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    


input_data = {
    'ram': 8, 'ram_expandable': 8, 'ghz': 2.5, 'display': 15.6, 'ssd': 512, 'hdd': 8,
    'processor_brand': 'Intel', 'ram_type': 'DDR4', 'display_type': 'LCD', 'gpu_brand': 'Intel'
}

input_df = pd.DataFrame([input_data])

# 1. Create ALL expected columns (with 0s) FIRST and in correct order
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# 2. Perform one hot encoding on input data using the function
input_df_temp = one_hot_encode_fixed_categories(input_data)
for col in input_df_temp.columns:
    if col in input_df.columns:
        input_df[col] = input_df_temp[col]

numerical_cols = [ 'ram_expandable', 'ram', 'ghz', 'display', 'ssd', 'hdd']

# Scale AFTER one-hot encoding, on ALL relevant columns
cols_to_scale = numerical_cols + [col for col in feature_names if col not in numerical_cols]
df_to_scale = input_df[cols_to_scale]

scaled_data = loaded_scaler.transform(df_to_scale)

scaled_df = pd.DataFrame(scaled_data, index = input_df.index, columns= cols_to_scale)
input_df[cols_to_scale] = scaled_df

# Ensure final column order matches feature_names
input_df = input_df[feature_names]

prediction = loaded_model.predict(input_df)
print(f"Predicted price: {prediction[0]}")
print("Final Dataframe for prediction:\n", input_df)