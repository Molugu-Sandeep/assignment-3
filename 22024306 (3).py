import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler




# Read the dataset
df_1= pd.read_csv('API_NY.GDP.TOTL.RT.ZS_DS2_en_csv_v2_5363504.csv', skiprows=4)

# Display the head of the dataset
print(df_1.head())


import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("API_NY.GDP.TOTL.RT.ZS_DS2_en_csv_v2_5363504.csv", skiprows=4)

# Select relevant columns and drop missing values
df_cleaned = df[["Country Name", "2019"]].dropna()

# Rename columns
df_cleaned.columns = ["Country", "Total natural resources rents"]

# Set index to country name
df_cleaned.set_index("Country", inplace=True)

# Remove rows with invalid values (negative or zero)
df_cleaned = df_cleaned[df_cleaned["Total natural resources rents"] > 0]

# Log-transform the values to reduce skewness
df_cleaned["Total natural resources rents"] = np.log(df_cleaned["Total natural resources rents"])

# Standardize the data using z-score normalization
df_cleaned = (df_cleaned - df_cleaned.mean()) / df_cleaned.std()

# Save the cleaned dataset to a new file
df_cleaned.to_csv("cluster_dataset.csv")




df = pd.read_csv("API_NY.GDP.TOTL.RT.ZS_DS2_en_csv_v2_5363504.csv", skiprows=4)

# Select relevant columns and drop missing values
df_selected = df[["Country Name", "2019"]].dropna()

# Rename columns
df_selected.columns = ["Country", "Total Natural Resources Rents"]

# Set index to country name
df_selected.set_index("Country", inplace=True)

# Remove rows with invalid values (negative or zero)
df_cleaned = df_selected[df_selected["Total Natural Resources Rents"] > 0]

# Log-transform the values to reduce skewness
df_cleaned["Total Natural Resources Rents"] = np.log(df_cleaned["Total Natural Resources Rents"])

# Standardize the data using z-score normalization
df_normalized = (df_cleaned - df_cleaned.mean()) / df_cleaned.std()

# Save the cleaned dataset to a new file
df_normalized.to_csv("task1clean_dataset.csv")





# Load the cleaned dataset
df = pd.read_csv("task1clean_dataset.csv")
# Extract the column of interest and normalize the data
data = df['Total Natural Resources Rents'].values.reshape(-1, 1)
normalized_data = (data - data.mean()) / data.std()

# Define the range of number of clusters to try
num_clusters_range = range(2, 11)

# Iterate over the number of clusters and compute the silhouette score
silhouette_scores = []
for num_clusters in num_clusters_range:
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(normalized_data)
    silhouette_scores.append(silhouette_score(normalized_data, labels))

# Plot the silhouette scores
plt.plot(num_clusters_range, silhouette_scores)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal Number of Clusters")
plt.show()





# Extract Inflation column and normalize
X = df['Total Natural Resources Rents'].values.reshape(-1, 1)
X_norm = StandardScaler().fit_transform(X)

# Perform K-means clustering with n_clusters=3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_norm)
df['Cluster'] = kmeans.labels_

# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['red', 'green', 'blue']
for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    scatter = ax.scatter(cluster_data.index, cluster_data['Total Natural Resources Rents'],
                         color=colors[i], label=f'Cluster {i+1}')
plt.xticks(np.arange(0, df.shape[0], 50), np.arange(0, df.shape[0], 50), fontsize=12)
plt.xlabel('Country Index', fontsize=14)
plt.ylabel('Total Natural Resources Rents', fontsize=14)
plt.title('K-means Clustering Plot', fontsize=16)
ax.legend(fontsize=12)

# Add annotation for the cluster centers
centers = kmeans.cluster_centers_
for i, center in enumerate(centers):
    ax.annotate(f'Cluster {i+1} center: {center[0]:,.2f}', xy=(i+1, center[0]), xytext=(6, 0),
                textcoords="offset points", ha='left', va='center', fontsize=12, color=colors[i])

plt.show()


# Print the cluster members
for i in range(3):
    cluster_members = df[df['Cluster'] == i]
    print(f'Cluster {i+1} Members:')
    print(cluster_members)




# Read the dataset
df_2 = pd.read_csv('API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_5358346.csv', skiprows=4)

# Display the head of the dataset
df_2.head()




# Load the dataset into a pandas DataFrame
df = pd.read_csv('API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_5358346.csv', skiprows=4)

# Select only the necessary data for fitting analysis
df = df[['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', *df.columns[-32:-1]]]

# Rename columns to simpler names
df.columns = ['FitCountry', 'FitCode', 'IndicatorName', 'IndicatorCode', *range(1990, 2021)]  

# Melt the DataFrame to transform the columns into rows
df_melted = pd.melt(df, id_vars=['FitCountry', 'FitCode', 'IndicatorName', 'IndicatorCode'], var_name='Year', value_name='Value') 

# Drop rows with missing values
df_cleaned = df_melted.dropna()  # Remove rows with missing values

# Save the cleaned data to a new CSV file
df_cleaned.to_csv('task2clean_data.csv', index=False)  




# Load the cleaned dataset
df = pd.read_csv('task2clean_data.csv')

# Select the four European countries of interest
countries = ['France', 'Germany', 'Italy', 'Spain']
df_filtered = df[df['FitCountry'].isin(countries) & (df['Year'] == 2020)]

# Plot the GDP growth rate for each country
plt.figure(figsize=(12, 8))
for country in countries:
    country_data = df_filtered[df_filtered['FitCountry'] == country]
    plt.bar(country_data['FitCountry'], country_data['Value'], label=country)

plt.xlabel('Country', fontsize=14)
plt.ylabel('GDP Growth Rate', fontsize=14)
plt.title('GDP Growth Rate in European Countries (2020)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

""" Module errors. It provides the function err_ranges which calculates upper
and lower limits of the confidence interval. """

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   


# Load the cleaned dataset
df = pd.read_csv('task2clean_data.csv')

# Filter data for France
france_data = df[df['FitCountry'] == 'France']

# Extract the necessary columns
years = france_data['Year'].values
values = france_data['Value'].values

# Fit a polynomial curve to the data
coeffs = np.polyfit(years, values, deg=2)
poly_func = np.poly1d(coeffs)

# Calculate the residuals
residuals = values - poly_func(years)

# Calculate the standard deviation of the residuals
std_dev = np.std(residuals)

# Generate predictions for future years
future_years = np.arange(years.min(), years.max() + 21)  # Predict for 20 additional years
predicted_values = poly_func(future_years)

# Calculate upper and lower confidence bounds
upper_bound = predicted_values + 2 * std_dev
lower_bound = predicted_values - 2 * std_dev

# Plot the best fitting function and confidence range
plt.figure(figsize=(12, 8), dpi=80)  # Set fixed dimensions for the figure
plt.plot(years, values, 'ko', label='Actual Data')
plt.plot(future_years, predicted_values, 'r-', label='Best Fitting Function')
plt.fill_between(future_years, lower_bound, upper_bound, color='yellow', alpha=0.4, label='Confidence Range')
plt.xlabel('Year', fontsize=14)
plt.ylabel('GDP growth (annual %)', fontsize=14)
plt.title('Polynomial Model Fit for France', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
