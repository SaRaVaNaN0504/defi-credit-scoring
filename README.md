# DeFi Wallet Credit Scoring Model

This project implements a one-step script to analyze raw, transaction-level data from the Aave V2 protocol. It engineers features from on-chain behavior to assign a credit score between 0 and 1000 to each unique wallet address.

## Project Overview

The primary goal is to create a robust and transparent credit scoring system based solely on a user's historical transaction data. Higher scores indicate reliable financial behavior, while lower scores reflect riskier patterns, such as having been liquidated. The model processes a raw JSON file containing 100,000 transactions and generates a final CSV with scores for all 3,497 unique wallets found in the dataset.

---

## Project Structure

The repository is organized as follows:



---

## Methodology and Architecture

The process is executed by a single script, `score_wallets.py`, which follows a multi-stage pipeline.

### 1. Data Preprocessing

The script begins by loading the `user-wallet-transactions.json` file. Key preprocessing steps include:
- **Handling Nested JSON**: The `actionData` column, which contains nested JSON objects with critical information, is normalized into separate columns (e.g., `amount`, `assetPriceUSD`).
- **Calculating USD Value**: The `amountUSD` for each transaction is calculated by multiplying the `amount` by the `assetPriceUSD`, as it was not provided directly.
- **Standardizing Columns**: Key columns like `userWallet` and `action` are renamed to `user` and `type` for consistent use throughout the pipeline.

### 2. Feature Engineering

Once the data is clean, the script engineers a set of features for each unique wallet to model its on-chain behavior:
- **`wallet_age_days`**: The number of days between the wallet's first and last transaction.
- **`transaction_count`**: The total number of interactions with the protocol.
- **`total_volume_usd`**: The sum of the USD value of all transactions for the wallet.
- **Transaction Type Counts**: The number of times a wallet performed specific actions (e.g., `Deposit`, `Borrow`, `Repay`).
- **Transaction Type Volumes**: The total USD value for each transaction type (e.g., `Deposit_total_usd`).
- **`repay_to_borrow_ratio`**: The ratio of total USD repaid to total USD borrowed. A value >= 1 indicates responsible borrowing.
- **`redeem_to_deposit_ratio`**: The ratio of total USD redeemed to total USD deposited.

### 3. Credit Scoring Model

A weighted scoring model is used for its transparency and interpretability. Monetary features are log-transformed using `np.log1p` to handle wide value distributions. All features are then scaled to a 0-1 range using `MinMaxScaler`.

The final score is a weighted sum of these normalized features:

| Feature                 | Weight | Rationale                                        |
| ----------------------- | :----: | ------------------------------------------------ |
| **Deposit Volume**      | +20%   | High capital contribution indicates seriousness. |
| **Repay/Borrow Ratio**  | +25%   | The strongest indicator of reliability.          |
| **Wallet Age**          | +15%   | A long history implies stability and experience. |
| **Total Volume**        | +15%   | Overall activity and engagement with the protocol. |
| **Transaction Count**   | +10%   | A measure of user activity.                      |
| **Liquidation Event**   | -35%   | The strongest negative indicator of risk.        |

The resulting raw scores are scaled again to produce the final **0 to 1000** credit score.

---

## How to Run the Project

To replicate the results, follow these steps:

**1. Clone the repository:**
```bash
git clone <your-repo-url>
cd DEFI-CREDIT-SCORING
