import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_clean_data(file_path):
    """
    Load data from a CSV file, rename columns, drop unnecessary columns,
    and set 'Country Name' as the index.
    """
    df = pd.read_csv(file_path)

    column_naming = {'Series Name': 'Indicators', '2011 [YR2011]': '2011',
                     '2012 [YR2012]': '2012', '2013 [YR2013]': '2013',
                     '2014 [YR2014]': '2014', '2015 [YR2015]': '2015',
                     '2016 [YR2016]': '2016', '2017 [YR2017]': '2017',
                     '2018 [YR2018]': '2018', '2019 [YR2019]': '2019',
                     '2020 [YR2020]': '2020', '2021 [YR2021]': '2021'}
    df.rename(columns=column_naming, inplace=True)

    columns_to_drop = ['Series Code', 'Country Code']
    df_cleaned = df.drop(columns=columns_to_drop)

    df_cleaned.set_index('Country Name', inplace=True)

    return df, df_cleaned


def generate_summary_statistics(grouped_data):
    """
    Generate and print summary statistics for each group in the grouped data.
    """
    for group_name, group in grouped_data:
        print(f"\nSummary Statistics for '{group_name}':")
        numeric_columns = group.select_dtypes(include=np.number)
        non_numeric_columns = group.select_dtypes(exclude=np.number)
        summary_stats_numeric = numeric_columns.describe()
        print("Numeric Columns:")
        print(summary_stats_numeric)
        summary_stats_non_numeric = non_numeric_columns.describe()
        print("\nNon-Numeric Columns:")
        print(summary_stats_non_numeric)


def generate_and_print_matrix(grouped_data, matrix_function, matrix_name):
    """
    Generate and print a matrix (correlation or covariance) 
    for each group in the grouped data.
    """
    for group_name, group in grouped_data:
        print(f"\n{matrix_name} Matrix for '{group_name}':")
        numeric_group = group.apply(pd.to_numeric, errors='coerce')
        numeric_group = numeric_group.dropna(axis=1, how='any')
        if len(numeric_group.columns) > 1:
            matrix_np = matrix_function(numeric_group.values)
            print(matrix_np)
        else:
            print("Insufficient data for matrix calculation.")


# Load original and cleaned data
df_stat, df_stat_cleaned = load_and_clean_data('stats_data.csv')

# Drop 'Series Code' and 'Country Code' from the original DataFrame
df_stat.drop(['Series Code', 'Country Code'], axis=1, inplace=True)

# Transpose the original DataFrame
df_stat_transposed = df_stat.set_index('Indicators').transpose()

# Drop NaN values from the cleaned DataFrame
df_stat_cleaned = df_stat_cleaned.dropna()

# Group by the 'Indicators' column in the cleaned data
grouped_by_indicator = df_stat_cleaned.groupby('Indicators')

# Generate and print summary statistics
generate_summary_statistics(grouped_by_indicator)

# Define functions for correlation and covariance matrices


def calculate_correlation_matrix(data_frame):
    """
    Generates the correlation matrix for the DataFrame. 
    """
    return np.corrcoef(data_frame, rowvar=False)


def calculate_covariance_matrix(data_frame):
    """
    Generates the Covariance Matrix for the DataFrame
    """
    return np.cov(data_frame, rowvar=False)


# Generate and print correlation matrices
generate_and_print_matrix(grouped_by_indicator,
                          calculate_correlation_matrix, 'Correlation')

# Generate and print covariance matrices
generate_and_print_matrix(grouped_by_indicator,
                          calculate_covariance_matrix, 'Covariance')

# Function to plot line graphs

    
def plot_line_graphs(df, selected_years, title):
    """
    Plot line graphs for each country over the selected years.
    """    
    plt.figure(figsize=(10, 6))
    sns.set(style="white")  # Set grid style to white

    for country in df.index:
        # Convert to float, excluding non-numeric values
        numeric_values = pd.to_numeric(df.loc[country], errors='coerce')
        plt.plot(df.columns, numeric_values, marker='o', label=country)

    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Country")  # Use the indicator name as the y-axis label
    plt.legend(title="Country", loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_grouped_bar_chart(df, selected_years, title, ylabel):
    """
    Plot grouped bar chart for selected years.
    """
    plt.figure(figsize=(12, 6))
    sns.set(style="white")  # Set grid style to white

    numeric_columns = df.columns[df.columns.isin(selected_years)]

    if len(numeric_columns) == 0:
        print("No numeric columns for the selected years.")
        return

    bar_width = 0.15
    positions = np.arange(len(df.index))

    for i, year in enumerate(selected_years):
        plt.bar(positions + i * bar_width,
                df[year].astype(float).values, width=bar_width, alpha=0.7, 
                label=year)

    plt.title(title)
    plt.xlabel("Country")
    plt.ylabel(ylabel)
    plt.xticks(positions + bar_width * (len(selected_years) - 1) /
               2, df.index, rotation=20, ha='right')
    plt.legend(title="Year", loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()


def generate_correlation_heatmap(df, selected_country, selected_indicators):
    # Filter the data for the selected country
    selected_country_data = df.loc[selected_country]

    # Filter the DataFrame for the selected indicators
    selected_country_data = selected_country_data[
        selected_country_data['Indicators'].isin(
        selected_indicators)]

    # Drop non-numeric columns
    numeric_columns = selected_country_data.select_dtypes(include=np.number)

    # Reset the index before creating the heatmap
    numeric_columns_reset = numeric_columns.reset_index(drop=True)

    # Remove the index and column names
    numeric_columns_reset.columns.name = None
    numeric_columns_reset.index.name = None

    # Check if there are enough numeric columns for correlation calculation
    if len(numeric_columns_reset.columns) > 1:
        # Calculate the correlation matrix for the selected country's indicators
        correlation_matrix = numeric_columns_reset.T.corr()

        # Create a correlation heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                    xticklabels=selected_indicators, 
                    yticklabels=selected_indicators)
        plt.title(f'Correlation Heatmap for {selected_country}')
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=20, ha='right')
        plt.show()


# Select years for line plot
selected_years_line_plot = ['2011', '2012', '2013',
                            '2014', '2015', '2016', 
                            '2017', '2018', '2019', '2020']

# Select the indicators
population_total = grouped_by_indicator.get_group('Population, total')
gov_effectiveness = grouped_by_indicator.get_group(
    'Government Effectiveness: Estimate')
net_migration = grouped_by_indicator.get_group('Net migration')
ghg_emissions = grouped_by_indicator.get_group(
    'Total greenhouse gas emissions (kt of CO2 equivalent)')

# Exclude the 'Indicators' column
population_total = population_total.drop(columns='Indicators')
gov_effectiveness = gov_effectiveness.drop(columns='Indicators')
net_migration = net_migration.drop(columns='Indicators')
ghg_emissions = ghg_emissions.drop(columns='Indicators')

# Plot line graphs
plot_line_graphs(population_total, selected_years_line_plot,
                 "Line Graph for Population, total (2011-2021)")
plot_line_graphs(gov_effectiveness, selected_years_line_plot,
                 "Line Graph for Government Effectiveness: Estimate (2011-2021)")
plot_line_graphs(net_migration, selected_years_line_plot,
                 "Line Graph for Net migration (2011-2021)")
plot_line_graphs(ghg_emissions, selected_years_line_plot, 
                 "Line Graph for Total Greenhouse Gas Emissions (kt of CO2 equivalent) (2011-2021)")

# Select years for bar plot
selected_years_bar_plot = ['2011', '2013', '2015', '2017', '2019']

# Select indicators for reference
forest_area_percentage = grouped_by_indicator.get_group(
    "Forest area (% of land area)")
population_density = grouped_by_indicator.get_group(
    "Population density (people per sq. km of land area)")

# Exclude the 'Indicators' column
forest_area_percentage = forest_area_percentage.drop(columns='Indicators')
population_density = population_density.drop(columns='Indicators')

# Plot grouped bar chart for "Population Density"
plot_grouped_bar_chart(population_density, selected_years_bar_plot,
                       "Population Density by Country for Selected Years", 
                       "Population Density (people per sq. km of land area)")

# Plot grouped bar chart for "Forest area (% of land area)"
plot_grouped_bar_chart(forest_area_percentage, selected_years_bar_plot,
                       "Forest Area by Country for Selected Years", 
                       "Forest Area (% of Land Area)")

# Select Countries to plot Heatmap
selected_country_1 = 'India'
selected_country_2 = 'South Africa'
selected_country_3 = 'Nigeria'
selected_country_4 = 'United States'
selected_country_5 = 'China'

# Select Indicators for the Heatmap
selected_indicators = ['Forest area (% of land area)', 'Forest area (sq. km)', 
                       'Population density (people per sq. km of land area)',
                       'Government Effectiveness: Estimate', 'Net migration',
                       'Population, total',
                       'Total greenhouse gas emissions (kt of CO2 equivalent)']

# Plot the HeatMap
generate_correlation_heatmap(
    df_stat_cleaned, selected_country_1, selected_indicators)
generate_correlation_heatmap(
    df_stat_cleaned, selected_country_2, selected_indicators)
generate_correlation_heatmap(
    df_stat_cleaned, selected_country_3, selected_indicators)
generate_correlation_heatmap(
    df_stat_cleaned, selected_country_4, selected_indicators)
generate_correlation_heatmap(
    df_stat_cleaned, selected_country_5, selected_indicators)