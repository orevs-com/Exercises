# Importing libraries necessary for the code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm


def read_and_clean_csv(csv_data, columns_to_drop=None, year_columns=None):
    """
    Read a CSV file, drop specified columns, and convert specified 
    year columns to numeric data.

    Parameters:
    - csv_data (str): Path to the CSV file.
    - columns_to_drop (list): List of columns to drop.
    - year_columns (list): List of columns to convert to numeric data.

    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    # Read the CSV file
    file = pd.read_csv(csv_data)

    # Drop specified columns
    if columns_to_drop:
        file = file.drop(columns=columns_to_drop, errors='ignore')

    # Convert specified year columns to numeric data
    if year_columns:
        file[year_columns] = file[year_columns].apply(
            pd.to_numeric, errors='coerce')

    return file


def select_data_for_scatter(data, indicators):
    """
    Select data for scatter plot based on a list of indicators.

    Parameters:
    - data (pd.DataFrame): DataFrame containing indicator data.
    - indicators (list): List of indicators to include in the scatter plot.

    Returns:
    - pd.DataFrame: DataFrame containing selected data for scatter plot.
    """
    scatter_data = data[data['Indicators'].isin(indicators)]
    return scatter_data


def normalize_and_kmeans(data, x_indicator, y_indicator, num_clusters=3):
    """
    Normalize the data, perform k-means clustering, and add 'Cluster' column.

    Parameters:
    - data (pd.DataFrame): DataFrame containing indicator data.
    - x_indicator (str): Name of the first indicator.
    - y_indicator (str): Name of the second indicator.
    - num_clusters (int): Number of clusters for k-means.

    Returns:
    - pd.DataFrame: DataFrame with added 'Cluster' column.
    - sklearn.cluster.KMeans: Fitted KMeans model.
    """
    common_countries = set(data[data['Indicators'] == x_indicator]
                           ['Country Name']).intersection(
        set(data[data['Indicators'] == y_indicator]['Country Name'])
    )

    xy_values = data[(data['Indicators'] == x_indicator) & (
        data['Country Name'].isin(common_countries))][['2010', 'Country Name']]
    xy_values = xy_values.rename(columns={'2010': 'x'})

    y_values = data[(data['Indicators'] == y_indicator) & (
        data['Country Name'].isin(common_countries))][['2010', 'Country Name']]
    y_values = y_values.rename(columns={'2010': 'y'})

    # Merge and drop NaN values
    normalized_data = pd.merge(
        xy_values, y_values, on='Country Name', how='inner').dropna()

    scaler = MinMaxScaler()
    normalized_data[['x', 'y']] = scaler.fit_transform(
        normalized_data[['x', 'y']])

    features = normalized_data[['x', 'y']]

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    normalized_data['Cluster'] = kmeans.fit_predict(features)

    return normalized_data, kmeans


def plot_kmeans_clusters(data, kmeans_model):
    """
    Plot a scatterplot with k-means clusters and cluster centers.

    Parameters:
    - data (pd.DataFrame): DataFrame containing indicator data.
    - kmeans_model (sklearn.cluster.KMeans): Fitted KMeans model.

    Returns:
    - Displays the scatterplot.
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        data['x'], data['y'], c=data['Cluster'], cmap='viridis', 
        edgecolors='k', s=50)

    # Plot cluster centers
    cluster_centers = kmeans_model.cluster_centers_
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                c='red', marker='X', s=200, label='Cluster Centers')

    plt.title('Country Clustering for CO2 Emission and Forest Area (2010)')
    plt.xlabel('Forest area')
    plt.ylabel('CO2 emissions')

    # Adding a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          label=f'Cluster {i}',
                          markerfacecolor=scatter.cmap(scatter.norm(i)), 
                          markersize=10) for i in 
               range(len(data['Cluster'].unique()))]
    handles.append(plt.Line2D([0], [0], marker='X', color='w', 
                              label='Cluster Centers',
                              markerfacecolor='red', markersize=10))

    plt.legend(handles=handles, title="Clusters")
    plt.show()


def calculate_correlation_and_heatmap(data, indicators):
    """
    Calculate correlation matrix and display a heatmap.

    Parameters:
    - data (pd.DataFrame): DataFrame containing indicator data.
    - indicators (list): List of indicators for correlation.

    Returns:
    - Displays the heatmap.
    """
    correlation_matrix = data[data['Indicators'].isin(indicators)].pivot(
        index='Country Name', columns='Indicators', values='2010').corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('Correlation Heatmap')
    plt.show()


def plot_line_for_each_country(data, indicator_name, projection_years=2040):
    """
    Create a separate line plot for each country based on the indicator, 
    including projections until 2040 with corrected confidence intervals.

    Parameters:
    - data (pd.DataFrame): Cleaned DataFrame.
    - indicator_name (str): Name of the indicator to plot.
    - projection_years (int): End year for projections.

    Returns:
    - Displays the plots.
    """
    indicator_data = data[data['Indicator'] == indicator_name]

    for index, row in indicator_data.iterrows():
        country_name = row['Country Name']
        years = np.array([int(year) for year in row.index[2:]], dtype=float)
        values = np.array(row[2:], dtype=float)

        # Curve fitting using a simple polynomial fit
        degree = 2
        coefficients = np.polyfit(years, values, degree)

        # Extend the years to include projections
        extended_years = np.concatenate(
            [years, np.arange(years[-1] + 1, projection_years + 1)])

        # Project values using the fitted curve
        projected_values = np.polyval(coefficients, extended_years)

        # Use statsmodels to calculate confidence interval
        x_vals = sm.add_constant(years)
        x_pred = sm.add_constant(extended_years)

        model = sm.OLS(values, x_vals)
        fitted_model = model.fit()

        prediction = fitted_model.get_prediction(x_pred)
        conf_int = prediction.conf_int()

        lower_bound = conf_int[:, 0]
        upper_bound = conf_int[:, 1]

        plt.figure(figsize=(12, 5))
        plt.plot(extended_years, projected_values,
                 label='Projection', linestyle='--', color='red')
        plt.fill_between(extended_years, lower_bound, upper_bound,
                         color='lightgray', label='Confidence Interval')
        plt.plot(years, values, label='CO2 Emission',
                 linestyle='-', color='blue')
        plt.title(f'{country_name} -- CO2 Emission Over Time ')
        plt.xlabel('Year')
        plt.ylabel('CO2 Emission')
        plt.legend(loc='best')
        plt.xticks(rotation=45, ha='right', ticks=extended_years[::5])
        plt.tight_layout()
        plt.show()


# Read the CSV files
cluster_data = read_and_clean_csv('2010_data.csv', columns_to_drop=[
                                  'Series Code', 'Country Code'], 
    year_columns=['2010'])

fitting_data = read_and_clean_csv('country_fit.csv', columns_to_drop=[
                                  'Series Code', 'Country Code'], 
    year_columns=[str(year) for year in range(1990, 2021)])

# Display the head of the "cluster_data" dataframe
print("Head of the 'cluster_data' dataframe:")
print(cluster_data.head())

# Display the head of the "fitting_data" dataframe
print("\nHead of the 'fitting_data' dataframe:")
print(fitting_data.head())

# Calculate correlation and display heatmap for cluster_data
indicators_for_correlation = ["CO2 emissions (kt)", "Population, total",
                              "Forest area (sq. km)", 
                              "Agricultural land (sq. km)", 
                              "GDP growth (annual %)"]
calculate_correlation_and_heatmap(cluster_data, indicators_for_correlation)

# Select data for scatter plot from cluster_data
scatter_data = select_data_for_scatter(
    cluster_data, ["Forest area (sq. km)", "CO2 emissions (kt)"])

# Display the head of the scatter dataframe
print("\nHead of the scatter dataframe:")
print(scatter_data.head())

# Normalize the data and perform k-means clustering on cluster_data
normalized_and_clustered_data, kmeans_model = normalize_and_kmeans(
    cluster_data, "Forest area (sq. km)", "CO2 emissions (kt)", num_clusters=3)

# Plot the scatterplot using the modified plot_scatter function
plot_kmeans_clusters(normalized_and_clustered_data, kmeans_model)

# Filter countries in Cluster 0 from cluster_data
cluster_0_countries = normalized_and_clustered_data[
    normalized_and_clustered_data['Cluster'] == 0]

# Set the option to display all rows without truncation
pd.set_option('display.max_rows', None)

# Display all columns for countries in Cluster 0 from cluster_data
print("Countries in Cluster 0:")
print(cluster_0_countries)

# Plot line for each country based on CO2 emissions from fitting_data
indicator_name = 'CO2 emissions (kt)'
plot_line_for_each_country(fitting_data, indicator_name, projection_years=2040)
