# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:38:39 2023

@author: Dell

"""
""" 
This is a code to show the line plot of net migration of five countries 
over a period of five years (2016-2020)
"""

# Importing Pandas, Numpy and Matplotlib

# Creating DataFrame for csv file "migration"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
migration = pd.read_csv('migration.csv')
print(migration)

# Selecting the Five Countries to be analyzed
migration_five = migration.iloc[[13, 29, 109, 174, 251],
                                [0, 60, 61, 62, 63, 64]]

# Reseting Index of "migration_five" DataFrame
migration_five = migration_five.reset_index(drop=True)
print(migration_five)

# Define function to create lineplot


def lineplot():
    """ 
    Function to create a lineplot. The years are defined on the x-axis and
    the migration data are defined on the y-axis
    """

    # Selecting the columns (years) for plotting
    years = migration_five.columns[1:]

    # Plotting data for each country
    for index, row in migration_five.iterrows():
        country_name = row['Country Name']
        migration_data = row[years].astype(float)  # Convert to numeric data
        plt.plot(years, migration_data, label=country_name)

    # Display labels and title
    plt.xlabel('Year')
    plt.ylabel('Migration Data')
    plt.title('Net Migration Data Over Time (2016-2020)')

    # Display legend
    plt.legend()

    # Displaying the plot
    plt.show()


# Retun Function
lineplot()


""" 
This is a code to show the Bar plot of net migration of five 
countries over a period of five years (2016-2020). This shows the relational 
data of the five countries selected
"""


# Creating DataFrame for csv file "migration"
migration = pd.read_csv('migration.csv')
print(migration)

# Selecting the Five Countries to be analyzed
migration_five = migration.iloc[[13, 29, 109, 174, 251],
                                [0, 60, 61, 62, 63, 64]]

# Resetting Index of "migration_five" DataFrame
migration_five = migration_five.reset_index(drop=True)
print(migration_five)

# Define function to create Bar Plot


def barplot():

    # Selecting the columns (years) for plotting
    years = migration_five.columns[1:]

    # Creating a bar plot for each country's migration data per year
    for index, row in migration_five.iterrows():
        country_name = row['Country Name']
        migration_data = row[years].astype(float)
        x = np.arange(len(years))  # the label locations
        width = 0.2  # the width of the bars

        plt.bar(x + (index-2)*width, migration_data, width, label=country_name)

    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('Migration Data')
    plt.title('Net Migration Data Per Country Per Year')

    # Adding legend
    plt.legend()

    # Adding x-axis labels for years
    plt.xticks(np.arange(len(years)), years)

    # Displaying the plot
    plt.show()


# Return Barplot function
barplot()

""" 
This is a code to show the pie plot of totL net migration of five countries 
over a period of five years (2016-2020)
"""


def migration_pie():
    # Creating DataFrame for csv file "migration"
    migration = pd.read_csv('migration.csv')
    print(migration)

    # Selecting the Five Countries to be analyzed
    migration_five = migration.iloc[[13, 29, 109, 174, 251],
                                    [0, 60, 61, 62, 63, 64]]

    # Resetting Index of "migration_five" DataFrame
    migration_five = migration_five.reset_index(drop=True)
    print(migration_five)

    year = ['2016', '2017', '2018', '2019', '2020']
    migration_five['Total Net Migration'] = migration_five[year].sum(
        axis=1).abs()
    print(migration_five)

    # Creating a pie chart for total net migration
    plt.figure(figsize=(10, 10))
    plt.pie(migration_five['Total Net Migration'],
            labels=migration_five['Country Name'],
            autopct=('%d%%'), labeldistance=1.02)
    plt.axis('equal')
    plt.title('Total Net Migration Over the Years per Country (2016-2020)')
    plt.show()


# Call the function to generate the pie chart
migration_pie()
