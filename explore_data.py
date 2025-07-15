# explore_data.py
import pandas as pd

# Load the dataset
try:
    df = pd.read_json('user-wallet-transactions.json')

    # Print basic information
    print("Data Head:")
    print(df.head())
    print("\nData Info:")
    df.info()

    # Check for different transaction types
    print("\nTransaction Types:")
    print(df['type'].unique())

except FileNotFoundError:
    print("Error: 'user_transactions.json' not found. Please download it to the project directory.")
except Exception as e:
    print(f"An error occurred: {e}")