import pandas as pd
import matplotlib.pyplot as plt

def plot_age_histogram(csv_file, age_column="Age", bins=10):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    
    # Drop missing values in the age column
    age_data = df[age_column].dropna()

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(age_data, bins=bins, edgecolor='black')
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_age_histogram('Cleaned_data.csv')
