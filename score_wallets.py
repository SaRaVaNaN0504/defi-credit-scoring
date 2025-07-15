import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
from datetime import datetime

def preprocess_data(df):
    """
    Normalizes nested data, calculates amountUSD, and standardizes column names.
    """
    # Debug: Print initial columns and data types
    print("Initial columns:", df.columns.tolist())
    print("Initial data types:")
    print(df.dtypes)
    
    # Handle MongoDB-style objects in _id, createdAt, updatedAt
    if '_id' in df.columns:
        df['_id'] = df['_id'].apply(lambda x: x.get('$oid', str(x)) if isinstance(x, dict) else str(x))
    
    if 'createdAt' in df.columns:
        df['createdAt'] = df['createdAt'].apply(lambda x: x.get('$date', str(x)) if isinstance(x, dict) else str(x))
    
    if 'updatedAt' in df.columns:
        df['updatedAt'] = df['updatedAt'].apply(lambda x: x.get('$date', str(x)) if isinstance(x, dict) else str(x))
    
    # 1. Normalize the 'actionData' column which contains nested JSON
    try:
        if 'actionData' in df.columns:
            print("Normalizing actionData...")
            
            # First, let's examine the actionData structure
            sample_action_data = df['actionData'].iloc[0]
            print(f"Sample actionData: {sample_action_data}")
            
            # Normalize the actionData
            action_data_normalized = pd.json_normalize(df['actionData'])
            print("ActionData normalized columns:", action_data_normalized.columns.tolist())
            
            # Drop the original 'actionData' to prevent conflicts
            df = df.drop(columns=['actionData'])
            
            # Join the normalized data
            df = pd.concat([df, action_data_normalized], axis=1)
            
        else:
            print("Warning: 'actionData' column not found in DataFrame")
    except Exception as e:
        print(f"Critical Error: Could not normalize 'actionData'. Error: {e}")
        sys.exit(1)

    # Debug: Print columns after normalization
    print("Columns after normalization:", df.columns.tolist())
    
    # Remove any duplicate columns that might have been created
    df = df.loc[:, ~df.columns.duplicated()]
    print("Columns after removing duplicates:", df.columns.tolist())
    
    # 2. Rename core columns for consistency
    if 'userWallet' in df.columns:
        df.rename(columns={'userWallet': 'user'}, inplace=True)
    
    # Handle the 'action' column renaming to 'type'
    if 'action' in df.columns:
        df.rename(columns={'action': 'original_action'}, inplace=True)  # Keep original as backup
    
    # Debug: Check if 'type' column exists and its content
    if 'type' in df.columns:
        print("'type' column found. Sample values:")
        print(df['type'].head(10))
        
        # Ensure type column is a Series
        if isinstance(df['type'], pd.DataFrame):
            print("Warning: 'type' column is a DataFrame, converting to Series")
            # If type is a DataFrame, take the first column
            df['type'] = df['type'].iloc[:, 0]
        
        # Convert to string and get unique values
        df['type'] = df['type'].astype(str)
        
        print("Unique values in type column:", df['type'].unique())
    else:
        # If type column doesn't exist, use the original action column
        print("Warning: 'type' column not found after normalization, using original action column")
        if 'original_action' in df.columns:
            df['type'] = df['original_action'].astype(str)
            print("Unique values in type column (from action):", df['type'].unique())
        else:
            print("Error: Neither 'type' nor 'action' columns found")
            sys.exit(1)
    
    # 3. *** CRITICAL STEP: Calculate amountUSD ***
    if 'amount' in df.columns and 'assetPriceUSD' in df.columns:
        # Ensure columns are numeric, coercing any errors to NaN
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['assetPriceUSD'] = pd.to_numeric(df['assetPriceUSD'], errors='coerce')
        # Fill any missing values with 0 before multiplying
        df['amount'] = df['amount'].fillna(0)
        df['assetPriceUSD'] = df['assetPriceUSD'].fillna(0)
        df['amountUSD'] = df['amount'] * df['assetPriceUSD']
    else:
        print("Critical Error: 'amount' or 'assetPriceUSD' column not found after processing.")
        print("Available columns:", df.columns.tolist())
        print("Cannot calculate transaction values.")
        sys.exit(1)

    # Check for essential columns after processing
    required_cols = ['user', 'type', 'timestamp', 'amountUSD']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns are: {df.columns.tolist()}")
        sys.exit(1)
    
    return df


def engineer_features(df):
    """
    Engineers features from the preprocessed transaction data.
    """
    # Convert timestamp to datetime objects if not already
    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Debug: Check the data before pivot operations
    print("Data shape before feature engineering:", df.shape)
    print("Sample data:")
    print(df[['user', 'type', 'amountUSD']].head())
    print("Unique transaction types:", df['type'].unique())
    print("Number of unique users:", df['user'].nunique())

    # Set up ID column if not present
    if 'id' not in df.columns:
        df['id'] = df.index

    # Aggregate features per user
    user_features = df.groupby('user').agg(
        transaction_count=('id', 'count'),
        first_transaction_date=('timestamp', 'min'),
        last_transaction_date=('timestamp', 'max'),
        total_volume_usd=('amountUSD', 'sum')
    ).reset_index()

    # Calculate wallet history in days
    user_features['wallet_age_days'] = (
        user_features['last_transaction_date'] - user_features['first_transaction_date']
    ).dt.days
    
    # Handle cases where first and last transaction are the same (single transaction)
    user_features['wallet_age_days'] = user_features['wallet_age_days'].fillna(0)

    # --- Transaction-specific features ---
    try:
        print("Creating transaction type count pivot table...")
        # Create pivot table for transaction type counts
        tx_type_counts = df.pivot_table(
            index='user', 
            columns='type', 
            values='id', 
            aggfunc='count', 
            fill_value=0
        )
        
        print("Transaction type counts columns:", tx_type_counts.columns.tolist())
        
        print("Creating transaction amount sum pivot table...")
        # Create pivot table for transaction amounts
        amount_sums = df.pivot_table(
            index='user', 
            columns='type', 
            values='amountUSD', 
            aggfunc='sum', 
            fill_value=0
        )
        
        # Rename amount columns to avoid conflicts
        amount_sums.columns = [f'{col}_total_usd' for col in amount_sums.columns]
        print("Amount sums columns:", amount_sums.columns.tolist())
        
    except Exception as e:
        print(f"Error during pivot table creation: {e}")
        print("DataFrame info:")
        print(df.info())
        print("Type column unique values:", df['type'].unique())
        print("Type column sample values:")
        print(df['type'].head(20))
        sys.exit(1)

    # Merge all features into a single DataFrame
    print("Merging features...")
    user_features = user_features.merge(tx_type_counts, on='user', how='left')
    user_features = user_features.merge(amount_sums, on='user', how='left')
    user_features = user_features.fillna(0)
    
    print("Final user features shape:", user_features.shape)
    print("Final user features columns:", user_features.columns.tolist())

    # --- Create insightful ratio features ---
    # Map common transaction types to standardized names
    type_mapping = {
        'repay': 'repay_total_usd',
        'borrow': 'borrow_total_usd', 
        'redeemunderlying': 'redeem_total_usd',
        'deposit': 'deposit_total_usd',
        'liquidationcall': 'liquidation_total_usd'
    }
    
    # Create ratio features based on available columns
    repay_col = None
    borrow_col = None
    redeem_col = None
    deposit_col = None
    
    for col in user_features.columns:
        if 'repay' in col.lower() and 'total_usd' in col.lower():
            repay_col = col
        elif 'borrow' in col.lower() and 'total_usd' in col.lower():
            borrow_col = col
        elif 'redeem' in col.lower() and 'total_usd' in col.lower():
            redeem_col = col
        elif 'deposit' in col.lower() and 'total_usd' in col.lower():
            deposit_col = col
    
    print(f"Found columns - Repay: {repay_col}, Borrow: {borrow_col}, Redeem: {redeem_col}, Deposit: {deposit_col}")
    
    # Calculate ratios
    if repay_col and borrow_col:
        user_features['repay_to_borrow_ratio'] = user_features[repay_col] / (user_features[borrow_col] + 1)
    else:
        user_features['repay_to_borrow_ratio'] = 0
        
    if redeem_col and deposit_col:
        user_features['redeem_to_deposit_ratio'] = user_features[redeem_col] / (user_features[deposit_col] + 1)
    else:
        user_features['redeem_to_deposit_ratio'] = 0
    
    return user_features

def calculate_credit_scores(user_features):
    """
    Calculates credit scores based on engineered features.
    """
    # Create a copy to avoid modifying the original
    features_copy = user_features.copy()
    
    # Print available columns for debugging
    print("Available columns for scoring:", features_copy.columns.tolist())
    
    # Define base weights for scoring
    base_weights = {
        'wallet_age_days': 0.15,
        'transaction_count': 0.10,
        'total_volume_usd': 0.15,
        'repay_to_borrow_ratio': 0.25,
    }
    
    # Look for liquidation-related columns
    liquidation_penalty = 0
    liquidation_cols = [col for col in features_copy.columns if 'liquidation' in col.lower()]
    
    if liquidation_cols:
        liquidation_col = liquidation_cols[0]
        print(f"Found liquidation column: {liquidation_col}")
        base_weights[liquidation_col] = -0.35  # Penalty for liquidations
    
    # Look for deposit-related columns
    deposit_cols = [col for col in features_copy.columns if 'deposit' in col.lower() and 'total_usd' in col.lower()]
    if deposit_cols:
        deposit_col = deposit_cols[0]
        print(f"Found deposit column: {deposit_col}")
        base_weights[deposit_col] = 0.20
    
    # Ensure all required features exist
    score_features = list(base_weights.keys())
    for feature in score_features:
        if feature not in features_copy.columns:
            features_copy[feature] = 0
            print(f"Warning: Feature '{feature}' not found, setting to 0")

    # Extract features for scoring
    X = features_copy[score_features]
    
    # Debug: Print feature statistics
    print("Feature statistics:")
    print(X.describe())
    
    # Handle any infinite or very large values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Apply log transformation to monetary amounts to handle large values
    monetary_cols = [col for col in X.columns if 'total_usd' in col.lower() or 'volume' in col.lower()]
    for col in monetary_cols:
        X[col] = np.log1p(X[col])  # log1p handles 0 values better

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate weighted scores
    weight_array = np.array([base_weights[feature] for feature in score_features])
    raw_scores = np.dot(X_scaled, weight_array)
    
    print(f"Raw scores - Min: {raw_scores.min():.4f}, Max: {raw_scores.max():.4f}, Mean: {raw_scores.mean():.4f}")

    # Scale scores to 0-1000 range
    score_scaler = MinMaxScaler(feature_range=(0, 1000))
    credit_scores = score_scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()
    
    # Create result DataFrame
    result_df = features_copy[['user']].copy()
    result_df['credit_score'] = credit_scores.astype(int)

    return result_df

def analyze_and_save_results(scored_wallets, user_features):
    """
    Performs analysis and saves all deliverables.
    """
    full_analysis_df = user_features.merge(scored_wallets, on='user')

    # Create score distribution plot
    plt.figure(figsize=(12, 7))
    plt.hist(full_analysis_df['credit_score'], bins=20, range=(0, 1000), edgecolor='black', alpha=0.7)
    plt.title('Distribution of Credit Scores', fontsize=16)
    plt.xlabel('Credit Score Range', fontsize=12)
    plt.ylabel('Number of Wallets', fontsize=12)
    plt.xticks(range(0, 1001, 100))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully saved score_distribution.png")

    # Analyze different score ranges
    low_scorers = full_analysis_df[full_analysis_df['credit_score'] < 200]
    medium_scorers = full_analysis_df[(full_analysis_df['credit_score'] >= 200) & (full_analysis_df['credit_score'] < 800)]
    high_scorers = full_analysis_df[full_analysis_df['credit_score'] >= 800]
    
    print(f"Number of low scorers (< 200): {len(low_scorers)}")
    print(f"Number of medium scorers (200-799): {len(medium_scorers)}")
    print(f"Number of high scorers (>= 800): {len(high_scorers)}")
    
    # Generate analysis markdown
    with open('analysis.md', 'w') as f:
        f.write("# DeFi Wallet Credit Score Analysis\n\n")
        f.write("This document provides an analysis of the wallet credit scores generated by the DeFi credit scoring model.\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"**Total wallets analyzed:** {len(full_analysis_df):,}\n\n")
        f.write(f"**Average credit score:** {full_analysis_df['credit_score'].mean():.2f}\n\n")
        f.write(f"**Median credit score:** {full_analysis_df['credit_score'].median():.2f}\n\n")
        f.write(f"**Standard deviation:** {full_analysis_df['credit_score'].std():.2f}\n\n")
        
        f.write("## Score Distribution\n\n")
        f.write("The following graph shows the distribution of credit scores across all wallets:\n\n")
        f.write("![Score Distribution](score_distribution.png)\n\n")
        
        f.write("## Score Categories\n\n")
        f.write(f"- **Low Risk (800-1000):** {len(high_scorers):,} wallets ({len(high_scorers)/len(full_analysis_df)*100:.1f}%)\n")
        f.write(f"- **Medium Risk (200-799):** {len(medium_scorers):,} wallets ({len(medium_scorers)/len(full_analysis_df)*100:.1f}%)\n")
        f.write(f"- **High Risk (0-199):** {len(low_scorers):,} wallets ({len(low_scorers)/len(full_analysis_df)*100:.1f}%)\n\n")
        
        if len(low_scorers) > 0:
            f.write("## High Risk Wallets (Score < 200)\n\n")
            f.write("High risk wallets typically exhibit:\n")
            f.write("- Higher likelihood of liquidations\n")
            f.write("- Poor repayment behavior\n")
            f.write("- Limited transaction history\n")
            f.write("- Lower overall transaction volumes\n\n")
            
            f.write("### Statistical Summary:\n\n")
            f.write("```\n" + low_scorers.describe().to_string() + "\n```\n\n")
        
        if len(high_scorers) > 0:
            f.write("## Low Risk Wallets (Score >= 800)\n\n")
            f.write("Low risk wallets demonstrate:\n")
            f.write("- Clean liquidation history\n")
            f.write("- Strong repayment behavior\n")
            f.write("- Established transaction patterns\n")
            f.write("- Higher transaction volumes\n\n")
            
            f.write("### Statistical Summary:\n\n")
            f.write("```\n" + high_scorers.describe().to_string() + "\n```\n\n")
        
        f.write("## Methodology\n\n")
        f.write("The credit score is calculated using the following factors:\n")
        f.write("- **Wallet Age (15%):** Time since first transaction\n")
        f.write("- **Transaction Count (10%):** Total number of transactions\n")
        f.write("- **Transaction Volume (15%):** Total USD value of transactions\n")
        f.write("- **Repayment Ratio (25%):** Ratio of repayments to borrows\n")
        f.write("- **Deposit Activity (20%):** Total deposit amounts\n")
        f.write("- **Liquidation Penalty (-35%):** Penalty for liquidation events\n\n")
        
        f.write("Scores are normalized to a 0-1000 scale where higher scores indicate lower risk.\n")
    
    print("Successfully generated analysis.md")

def main():
    """
    Main function to run the entire pipeline.
    """
    try:
        raw_df = pd.read_json('user-wallet-transactions.json')
        print(f"Successfully loaded {len(raw_df)} transactions")
        
        # Debug: Print sample data
        print("\nData Head:")
        print(raw_df.head())
        print("\nData Info:")
        print(raw_df.info())
        
        # Check transaction types
        if 'action' in raw_df.columns:
            print("\nTransaction Types:")
            print(raw_df['action'].value_counts())
        
    except FileNotFoundError:
        print("Error: 'user-wallet-transactions.json' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        sys.exit(1)

    # Preprocess the data
    processed_df = preprocess_data(raw_df)
    
    # Set up ID column
    if '_id' in processed_df.columns:
        processed_df.rename(columns={'_id': 'id'}, inplace=True)
    elif 'id' not in processed_df.columns:
        processed_df['id'] = processed_df.index

    print("\n--- Data Preprocessing Complete. Starting Feature Engineering... ---")
    
    user_features = engineer_features(processed_df)
    print("\n--- Feature Engineering Complete. Calculating Scores... ---")

    scored_wallets = calculate_credit_scores(user_features.copy())
    print("\n--- Score Calculation Complete. Generating Analysis... ---")

    analyze_and_save_results(scored_wallets, user_features)
    
    # Save results
    scored_wallets.to_csv('wallet_scores.csv', index=False)
    print("Successfully saved wallet_scores.csv")
    
    print("\n" + "="*50)
    print("EXECUTION SUMMARY")
    print("="*50)
    print(f"Total wallets processed: {len(scored_wallets):,}")
    print(f"Average credit score: {scored_wallets['credit_score'].mean():.2f}")
    print(f"Score range: {scored_wallets['credit_score'].min()} - {scored_wallets['credit_score'].max()}")
    
    # Score distribution
    score_ranges = [
        (0, 200, "High Risk"),
        (200, 500, "Medium-High Risk"),
        (500, 800, "Medium Risk"),
        (800, 1000, "Low Risk")
    ]
    
    for min_score, max_score, label in score_ranges:
        count = len(scored_wallets[(scored_wallets['credit_score'] >= min_score) & 
                                  (scored_wallets['credit_score'] < max_score)])
        percentage = (count / len(scored_wallets)) * 100
        print(f"{label} ({min_score}-{max_score-1}): {count:,} wallets ({percentage:.1f}%)")
    
    print("\nFiles generated:")
    print("- wallet_scores.csv: Individual wallet scores")
    print("- score_distribution.png: Score distribution visualization")
    print("- analysis.md: Detailed analysis report")
    print("\nProject execution complete!")

if __name__ == '__main__':
    main()