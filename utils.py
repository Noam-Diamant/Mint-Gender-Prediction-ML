"""
This module contains utilities, mappings and imports for the Mint gender prediction project.
"""
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, Dict
import xgboost as xgb


# sklearn imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score,
    roc_auc_score, log_loss, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# Constants
RANDOM_STATE = 42

# Category mapping
category_mapping = {
     # Main categories mapping to themselves
    'Entertainment': 'Entertainment',
    'Shopping': 'Shopping',
    'Personal Care': 'Personal Care',
    'Health & Fitness': 'Health & Fitness',
    'Kids': 'Kids',
    'Food & Dining': 'Food & Dining',
    'Gifts & Donations': 'Gifts & Donations',
    'Pets': 'Pets',
    'Education': 'Education',
    'Financial': 'Financial',
    'Travel': 'Travel',
    'Fees & Charges': 'Fees & Charges',
    'Business Services': 'Business Services',
    'Taxes': 'Taxes',
    'Bills & Utilities': 'Bills & Utilities',
    'Auto & Transport': 'Auto & Transport',
    'Transfer': 'Transfer',
    'Home': 'Home',
    'Investments': 'Investments',
    'Uncategorized': 'Uncategorized',
    'Income': 'Income',
    'Loans': 'Loans',

    # Entertainment
    'Arts': 'Entertainment',
    'Amusement': 'Entertainment',
    'Music': 'Entertainment',
    'Movies & DVDs': 'Entertainment',
    'Newspapers & Magazines': 'Entertainment',
    
    # Shopping
    'Clothing': 'Shopping',
    'Books': 'Shopping',
    'Electronics & Software': 'Shopping',
    'Hobbies': 'Shopping',
    'Sporting Goods': 'Shopping',
    
    # Personal Care
    'Hair': 'Personal Care',
    'Spa & Massage': 'Personal Care',
    'Laundry': 'Personal Care',
    
    # Health & Fitness
    'Dentist': 'Health & Fitness',
    'Doctor': 'Health & Fitness',
    'Eyecare': 'Health & Fitness',
    'Pharmacy': 'Health & Fitness',
    'Health Insurance': 'Health & Fitness',
    'Gym': 'Health & Fitness',
    'Sports': 'Health & Fitness',
    
    # Kids
    'Babysitter & Daycare': 'Kids',
    'Child Support': 'Kids',
    'Toys': 'Kids',
    'Kids Activities': 'Kids',
    'Allowance': 'Kids',
    'Baby Supplies': 'Kids',
    
    # Food & Dining
    'Groceries': 'Food & Dining',
    'Coffee Shops': 'Food & Dining',
    'Fast Food': 'Food & Dining',
    'Restaurants': 'Food & Dining',
    'Alcohol & Bars': 'Food & Dining',
    
    # Gifts & Donations
    'Gift': 'Gifts & Donations',
    'Charity': 'Gifts & Donations',
    
    # Pets
    'Pet Food & Supplies': 'Pets',
    'Pet Grooming': 'Pets',
    'Veterinary': 'Pets',
    
    # Education
    'Tuition': 'Education',
    'Student Loan': 'Education',
    'Books & Supplies': 'Education',
    
    # Financial
    'Life Insurance': 'Financial',
    'Financial Advisor': 'Financial',
    
    # Travel
    'Air Travel': 'Travel',
    'Hotel': 'Travel',
    'Rental Car & Taxi': 'Travel',
    'Vacation': 'Travel',
    
    # Fees & Charges
    'Service Fee': 'Fees & Charges',
    'Late Fee': 'Fees & Charges',
    'Finance Charge': 'Fees & Charges',
    'ATM Fee': 'Fees & Charges',
    'Bank Fee': 'Fees & Charges',
    'Trade Commissions': 'Fees & Charges',
    
    # Business Services
    'Advertising': 'Business Services',
    'Office Supplies': 'Business Services',
    'Printing': 'Business Services',
    'Shipping': 'Business Services',
    'Legal': 'Business Services',
    
    # Taxes
    'Federal Tax': 'Taxes',
    'State Tax': 'Taxes',
    'Local Tax': 'Taxes',
    'Sales Tax': 'Taxes',
    'Property Tax': 'Taxes',
    
    # Bills & Utilities
    'Television': 'Bills & Utilities',
    'Home Phone': 'Bills & Utilities',
    'Internet': 'Bills & Utilities',
    'Mobile Phone': 'Bills & Utilities',
    'Utilities': 'Bills & Utilities',
    
    # Auto & Transport
    'Gas & Fuel': 'Auto & Transport',
    'Parking': 'Auto & Transport',
    'Service & Parts': 'Auto & Transport',
    'Auto Payment': 'Auto & Transport',
    'Auto Insurance': 'Auto & Transport',
    'Public Transportation': 'Auto & Transport',
    
    # Transfer
    'Credit Card Payment': 'Transfer',
    'Transfer for Cash Spending': 'Transfer',
    
    # Home
    'Furnishings': 'Home',
    'Lawn & Garden': 'Home',
    'Home Improvement': 'Home',
    'Home Services': 'Home',
    'Home Insurance': 'Home',
    'Mortgage & Rent': 'Home',
    'Home Supplies': 'Home',
    
    # Investments
    'Deposit': 'Investments',
    'Withdrawal': 'Investments',
    'Dividend & Cap Gains': 'Investments',
    'Buy': 'Investments',
    'Sell': 'Investments',
    
    # Uncategorized
    'Cash & ATM': 'Uncategorized',
    'Check': 'Uncategorized',
    
    
    # Income
    'Paycheck': 'Income',
    'Returned Purchase': 'Income',
    'Bonus': 'Income',
    'Interest Income': 'Income',
    'Reimbursement': 'Income',
    'Rental Income': 'Income',
    
    # Loans
    'Loan Payment': 'Loans',
    'Loan Insurance': 'Loans',
    'Loan Principal': 'Loans',
    'Loan Interest': 'Loans',
    'Loan Fees and Charges': 'Loans'
}



# === Helper functions: leakage-safe feature builder ===

# Top 10 merchants (pre-computed to avoid data leakage)
TOP_10_MERCHANTS = [
    'Target', 'Amazon', 'Starbucks', 'Uber.com', 'Wal-Mart',
    'iTunes', 'McDonald\'s', 'CVS', 'Shell', 'Whole Foods', '7-Eleven', 'Lyft'
]

MAIN_CATEGORIES = [
    'Entertainment','Shopping','Personal Care','Health & Fitness','Kids',
    'Food & Dining','Gifts & Donations','Pets','Education','Financial',
    'Travel','Fees & Charges','Business Services','Taxes','Bills & Utilities',
    'Auto & Transport','Transfer','Home','Investments','Uncategorized',
    'Income','Loans'
]

DAYS = list(range(7))




def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0) else 0.0


#################################################################################################
# Visualization functions
#################################################################################################

def visualize(_df):
    # Build user-level features on all users
    X_all = build_user_features(_df)

    # Add extra features just for correlation analysis
    grp = _df.groupby('user_id', observed=True)
    
    # Calculate total spend (sum of negative amounts)
    total_spend = grp['amount'].apply(lambda x: abs(x[x < 0].sum())).reset_index(name='total_spend')
    X_all = X_all.merge(total_spend, on='user_id', how='left')
    
    # Calculate number of transactions
    num_transactions = grp.size().reset_index(name='num_transactions')
    X_all = X_all.merge(num_transactions, on='user_id', how='left')
    
    # Calculate amount standard deviation
    amount_std = grp['amount'].std().reset_index(name='amount_std')
    X_all = X_all.merge(amount_std, on='user_id', how='left')
    
    # Calculate positive ratio
    pos_ratio = grp['amount'].apply(lambda x: (x > 0).mean()).reset_index(name='pos_ratio')
    X_all = X_all.merge(pos_ratio, on='user_id', how='left')
    
    # Calculate fraction of income transactions
    frac_income_tx = grp['amount'].apply(lambda x: (x > 0).mean()).reset_index(name='frac_income_tx')
    X_all = X_all.merge(frac_income_tx, on='user_id', how='left')

    labels_unique = _df[['user_id', 'gender']].drop_duplicates(subset='user_id')
    X_all_l = X_all.merge(labels_unique, on='user_id', how='left')
    print("Money Flow analysis:")
    # Mean transactions per month by gender
    plt.figure(figsize=(5,4))
    sns.barplot(data=X_all_l, x='gender', y='transactions_per_month_mean', estimator=np.mean, errorbar=None)
    plt.title('Mean Transactions per Month by Gender')
    plt.ylabel('Mean Transactions per Month')
    plt.show()

    # Mean income and spend by gender
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    sns.barplot(data=X_all_l, x='gender', y='total_income', estimator=np.mean, errorbar=None, ax=axes[0])
    axes[0].set_title('Mean Total Income by Gender')
    axes[0].set_ylabel('Mean Total Income')
    
    sns.barplot(data=X_all_l, x='gender', y='total_spend', estimator=np.mean, errorbar=None, ax=axes[1])
    axes[1].set_title('Mean Total Spend by Gender')
    axes[1].set_ylabel('Mean Total Spend')
    plt.tight_layout()
    plt.show()

    print("Temporal analysis:")

    # Calculate number of active months
    _df['yyyymm'] = _df['date'].dt.to_period('M')  # Create yearmonth column first
    active_months = _df.groupby(['user_id', 'gender'])['yyyymm'].nunique().reset_index(name='num_active_months')

    # Mean active days and months by gender
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    sns.barplot(data=X_all_l, x='gender', y='active_days_per_month_mean', estimator=np.mean, errorbar=None, ax=axes[0])
    axes[0].set_title('Mean Active Days per Month by Gender')
    axes[0].set_ylabel('Mean Active Days per Month')
    
    sns.barplot(data=active_months, x='gender', y='num_active_months', estimator=np.mean, errorbar=None, ax=axes[1])
    axes[1].set_title('Mean Number of Active Months by Gender')
    axes[1].set_ylabel('Mean Active Months')
    plt.tight_layout()
    plt.show()

    # Fraction of transactions on each day of week by gender (mean per-user)
    dow_cols = [c for c in X_all.columns if c.startswith('dow_frac__')]
    dow_long = (
        X_all_l[['user_id', 'gender'] + dow_cols]
        .melt(id_vars=['user_id','gender'], var_name='dow', value_name='frac')
    )
    dow_long['dow'] = dow_long['dow'].str.replace('dow_frac__', '', regex=False).astype(int)
    dow_name_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
    dow_long['dow_name'] = dow_long['dow'].map(dow_name_map)

    plt.figure(figsize=(10,4))
    sns.barplot(data=dow_long, x='dow_name', y='frac', hue='gender', estimator=np.mean, errorbar=None)
    plt.title('Mean fraction of transactions by day-of-week (by gender)')
    plt.xlabel('day of week'); plt.ylabel('mean fraction')
    plt.tight_layout(); plt.show()

    # Weekday vs Weekend transactions
    _df['is_weekend'] = _df['dayofweek'].isin([5, 6])
    weekday_counts = _df.groupby(['user_id', 'is_weekend', 'gender']).size().reset_index(name='count')
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=weekday_counts, x='is_weekend', y='count', hue='gender')
    plt.title('Number of Transactions: Weekday vs Weekend by Gender')
    plt.xlabel('Is Weekend')
    plt.ylabel('Number of Transactions')
    plt.ylim(0, 2500)
    plt.show()

    # Monthly spending over time
    _df['yearmonth'] = _df['date'].dt.to_period('M')
    monthly_spend = _df.groupby(['yearmonth', 'gender', 'user_id'])['amount'].sum().reset_index()
    monthly_avg = monthly_spend.groupby(['yearmonth', 'gender'])['amount'].mean().reset_index()

    plt.figure(figsize=(15, 5))
    # Plot reference lines first
    plt.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label='$1000 threshold (March 2016)')
    plt.axhline(y=1300, color='red', linestyle='--', alpha=0.5, label='$1300 threshold (March 2016)')
    plt.axhline(y=1500, color='blue', linestyle='--', alpha=0.5, label='$1500 threshold (July 2016)') 
    plt.axhline(y=600, color='blue', linestyle='--', alpha=0.5, label='$600 threshold (July 2016)')
    plt.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='$0 threshold (Oct 2016)')
    plt.axhline(y=800, color='green', linestyle='--', alpha=0.5, label='$800 threshold (Oct 2016)')
    
    # Plot gender lines
    for gender in ['F', 'M']:
        gender_data = monthly_avg[monthly_avg['gender'] == gender]
        plt.plot(gender_data['yearmonth'].astype(str), gender_data['amount'], label=gender)
    plt.title('Average Monthly Spending Over Time by Gender')
    plt.xticks(rotation=45)
    plt.xlabel('Month')
    plt.ylabel('Average Amount')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Category analysis:")

    # Mean transaction fraction per category by gender
    cat_cols = [c for c in X_all.columns if c.startswith('cat_tx_frac__')]
    cat_long = (
        X_all_l[['user_id', 'gender'] + cat_cols]
        .melt(id_vars=['user_id','gender'], var_name='cat', value_name='tx_frac')
    )
    cat_long['cat'] = cat_long['cat'].str.replace('cat_tx_frac__', '', regex=False)
    plt.figure(figsize=(12,6))
    sns.barplot(data=cat_long, x='cat', y='tx_frac', hue='gender', estimator=np.mean, errorbar=None)
    plt.yscale('log')  # Set y-axis to log scale
    plt.title('Mean transaction fraction per category by gender (all users) - Log Scale')
    plt.xticks(rotation=90); plt.tight_layout(); plt.show()

    # Number of unique categories per user
    unique_cats = _df.groupby(['user_id', 'gender'])['category'].nunique().reset_index()
    plt.figure(figsize=(10, 5))
    sns.histplot(data=unique_cats, x='category', hue='gender', multiple="layer", stat='density')
    plt.title('Distribution of Unique Categories per User by Gender')
    plt.xlabel('Number of Unique Categories')
    plt.xscale('log')  # Set x-axis to log scale
    plt.show()

    print("Merchant analysis:")
    # Mean percentage of '-' descriptions by gender (pct of missing/placeholder merchant)
    # Compute directly from raw transactions to avoid relying on a stored feature
    dash_share = (
        _df.groupby('user_id', observed=True)['description']
           .apply(lambda s: (s == '-').mean())
           .reset_index(name='pct_dash_description')
    )
    dash_share = dash_share.merge(labels_unique, on='user_id', how='left')
    plt.figure(figsize=(5,4))
    sns.barplot(data=dash_share, x='gender', y='pct_dash_description', estimator=np.mean, errorbar=None)
    plt.title("Mean '-' description share by gender")
    plt.ylabel('mean pct_dash_description')
    plt.show()

    # Top 10 merchant descriptions by gender (counts)

    def top_n_merchants_by_gender(df_tx, n=10, drop_dash=True):
        out = {}
        for g in ['F', 'M']:
            gdf = df_tx[df_tx['gender'] == g]
            desc = gdf['description']
            if drop_dash:
                desc = desc[desc != '-']
            top = desc.value_counts().head(n).reset_index()
            top.columns = ['description', 'count']
            out[g] = top
        return out

    tops = top_n_merchants_by_gender(_df, n=10, drop_dash=True)

    # Combine data from both genders into one dataframe
    df_combined = pd.concat([
        tops['F'].assign(gender='F'),
        tops['M'].assign(gender='M')
    ])

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_combined, x='count', y='description', hue='gender', palette=['lightcoral', 'steelblue'])
    plt.title('Top 10 merchants by count and gender')
    plt.xlabel('count')
    plt.ylabel('merchant (description)')
    plt.tight_layout()
    plt.show()

    print("Correlation matrices:")
    # Correlation matrix of numeric features
    # All numeric features we want to analyze (if available)
    numeric_cols = [
        # Basic metrics
        'total_income', 'total_spend', 'num_transactions', 'amount_std',
        'amount_mean', 'amount_min', 'amount_max', 'pos_ratio',
        'frac_spend_tx', 'frac_income_tx', 'num_active_months',
        'transactions_per_month_mean', 'num_unique_descriptions',
        'active_days_per_month_mean', 'top_desc_share',
        
        # Category features
        'cat_tx_frac__Food & Dining', 'cat_tx_frac__Shopping',
        'cat_spend_share__Food & Dining', 'cat_spend_share__Shopping',
        'dow_frac__0', 'dow_frac__5', 'dow_frac__6'
    ]
    
    # Filter to only include columns that exist in the data
    available_cols = [col for col in numeric_cols if col in X_all_l.columns]
    corr_matrix = X_all_l[available_cols].corr()

    # Plot main correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of All Features')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Plot spending threshold correlation matrix
    threshold_cols = [
        'spend_over_1300_march_2016', 'spend_under_1000_march_2016',
        'spend_over_1500_july_2016', 'spend_under_600_july_2016',
        'spend_over_800_oct_2016', 'spend_under_0_oct_2016'
    ]
    threshold_corr = X_all_l[threshold_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(threshold_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Spending Threshold Features')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    

#################################################################################################
# Feature builder
#################################################################################################
def build_user_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build user-level features from a per-transaction DataFrame.
    The input must include only the transactions for the users we want to featurize
    (i.e., call per split to avoid leakage).

    Feature Groups and Aggregation Methods:

    1. Basic Transaction Counts:
        - num_unique_descriptions: Count of unique merchant descriptions

    2. Merchant/Description Features:
        - top_desc_share: Fraction of transactions with most frequent description
        - Binary indicators (1.0 if user has ever transacted with merchant, 0.0 otherwise):
        - has_{merchant}: One feature per top 10 merchant (TARGET, AMAZON, etc.)

    3. Amount-based Features:
        - amount_mean: Mean transaction amount
        - amount_min: Minimum transaction amount
        - amount_max: Maximum transaction amount
        - total_income: Sum of all positive amounts
        - frac_spend_tx: Fraction of negative amount transactions

    4. Category Distribution Features:
        For each main category, two features are computed:
        - cat_tx_frac__{category}: Fraction of transactions in category
        - cat_spend_share__{category}: Share of total spending in category
                                     (only negative amounts considered)

    5. Temporal Features:
        - transactions_per_month_mean: Average number of transactions per month
        - num_active_months: Total number of months with any activity
        - dow_frac__{0-6}: Fraction of transactions on each day of week
        - active_days_per_month_mean: Average number of unique days with activity per month
        - spend_over_1300_march_2016: Binary indicator if March 2016 spending > $1300
        - spend_under_1000_march_2016: Binary indicator if March 2016 spending < $1000
        - spend_over_1500_july_2016: Binary indicator if July 2016 spending > $1500  
        - spend_over_800_oct_2016: Binary indicator if October 2016 spending > $800
        - spend_under_600_july_2016: Binary indicator if July 2016 spending < $600
        - spend_under_0_oct_2016: Binary indicator if October 2016 spending < $0 (net income)

    Args:
        transactions: DataFrame with columns [user_id, description, category, amount, date]

    Returns:
        DataFrame with user_id index and all computed features as columns
    """
    tx = transactions.copy()

    # Base counts
    grp = tx.groupby('user_id', observed=True)
    features = pd.DataFrame(index=grp.size().index)

    #################################################################################################
    # Merchant/description aggregation
    #################################################################################################
    # Count how many unique merchant descriptions each user has
    features['num_unique_descriptions'] = grp['description'].nunique()

    # Top merchant share: fraction of most frequent description
    def top_desc_share(series: pd.Series) -> float:
        counts = series.value_counts(dropna=False)
        total = counts.sum()
        return counts.iloc[0] / total if total > 0 else 0.0
    features['top_desc_share'] = grp['description'].apply(top_desc_share)

    # Specific merchant indicators
    #################################################################################################
    def has_merchant(df_user: pd.DataFrame, merchant: str) -> float:
        """Returns 1.0 if user has ever transacted with merchant, 0.0 otherwise"""
        return float(df_user['description'].str.contains(merchant, case=False).any())

    # Track if user ever purchased at top 10 merchants
    for merchant in TOP_10_MERCHANTS:
        features[f'has_{merchant.lower().replace(" ", "_")}'] = grp.apply(lambda g: has_merchant(g, merchant))

    #################################################################################################
    # Amount-based Features
    #################################################################################################
    # Dictionary defining aggregation functions to apply to 'amount' column:
    agg_dict = {
        'amount': ['mean','min','max']
    }
    amount_stats = grp.agg(agg_dict)
    # Flatten columns
    amount_stats.columns = ['_'.join(col) for col in amount_stats.columns]
    features = features.join(amount_stats)
    
    # Calculate total income (sum of positive amounts)
    def calc_total_income(amounts: pd.Series) -> float:
        return amounts[amounts > 0].sum()
    
    features['total_income'] = grp['amount'].apply(calc_total_income)

    # Fraction of transactions that are spend
    def frac_spend(series: pd.Series) -> float:
        return (series < 0.0).mean()
    features['frac_spend_tx'] = grp['amount'].apply(frac_spend)

    #################################################################################################
    # Category aggregation
    #################################################################################################
    
    # Category distribution: fraction of transactions per main category
    # For each main category, calculate what percentage of a user's transactions fall into that category
    def frac_by_cat(s: pd.Series, cat: str) -> float:
        # Returns the fraction of values in series s that match the given category
        return (s == cat).mean()
    for cat in MAIN_CATEGORIES:
        # Create feature for each category showing what fraction of transactions are in that category
        features[f'cat_tx_frac__{cat}'] = grp['category'].apply(lambda s, c=cat: frac_by_cat(s, c))

    # Category distribution by spend share (only negative amounts)
    # For each main category, calculate what percentage of a user's total spending is in that category
    def spend_share_for_cat(df_user: pd.DataFrame, cat: str) -> float:
        # Calculate total amount spent in this category (negative amounts only)
        spent = df_user.loc[(df_user['category'] == cat) & (df_user['amount'] < 0.0), 'amount'].abs().sum()
        # Calculate total spending across all categories (negative amounts only) 
        total_spent = df_user.loc[df_user['amount'] < 0.0, 'amount'].abs().sum()
        # Return category spend as fraction of total spend, handling division by zero
        return _safe_div(spent, total_spent)
    for cat in MAIN_CATEGORIES:
        # Create feature for each category showing what fraction of total spend is in that category
        features[f'cat_spend_share__{cat}'] = grp.apply(lambda g, c=cat: spend_share_for_cat(g, c))

    #################################################################################################
    # Temporal aggregation
    #################################################################################################
    
    # Specific month spending thresholds
    # These features capture whether users exceed certain spending thresholds in specific months:
    # - spend_over_1300_march_2016: Binary indicator if March 2016 spending > $1300
    # - spend_over_1500_july_2016: Binary indicator if July 2016 spending > $1500  
    # - spend_over_800_oct_2016: Binary indicator if October 2016 spending > $800
    # - spend_under_600_july_2016: Binary indicator if July 2016 spending < $600
    # - spend_under_0_oct_2016: Binary indicator if October 2016 spending < $0 (net income)
    tx['yearmonth'] = tx['date'].dt.to_period('M')
    monthly_spend = tx.groupby(['user_id', 'yearmonth'])['amount'].sum().reset_index()
    
    # March 2016 spending thresholds
    march_2016_spend = monthly_spend[monthly_spend['yearmonth'] == pd.Period('2016-03')].set_index('user_id')['amount']
    features['spend_over_1300_march_2016'] = march_2016_spend.map(lambda x: float(x > 1300)).fillna(0.0)
    features['spend_under_1000_march_2016'] = march_2016_spend.map(lambda x: float(x < 1000)).fillna(0.0)
    
    # July 2016 spending > 1500
    july_2016_spend = monthly_spend[monthly_spend['yearmonth'] == pd.Period('2016-07')].set_index('user_id')['amount']
    features['spend_over_1500_july_2016'] = july_2016_spend.map(lambda x: float(x > 1500)).fillna(0.0)
    
    # October 2016 spending > 800
    oct_2016_spend = monthly_spend[monthly_spend['yearmonth'] == pd.Period('2016-10')].set_index('user_id')['amount']
    features['spend_over_800_oct_2016'] = oct_2016_spend.map(lambda x: float(x > 800)).fillna(0.0)
    
    # July 2016 spending < 600
    features['spend_under_600_july_2016'] = july_2016_spend.map(lambda x: float(x < 600)).fillna(0.0)
    
    # October 2016 spending < 0
    features['spend_under_0_oct_2016'] = oct_2016_spend.map(lambda x: float(x < 0)).fillna(0.0)
    # Calculate transactions per month
    monthly_tx_counts = tx.groupby(['user_id', 'yearmonth']).size().reset_index(name='count')
    features['transactions_per_month_mean'] = monthly_tx_counts.groupby('user_id')['count'].mean()

    # Temporal: fraction by day-of-week
    for d in DAYS:
        features[f'dow_frac__{d}'] = grp['dayofweek'].apply(lambda s, d=d: (s == d).mean())

    # Activity span in months and active days
    # Convert dates to year-month periods (e.g. 2016-12) and extract day
    tx['yyyymm'] = tx['date'].dt.to_period('M')
    tx['day'] = tx['date'].dt.day
    
    # Count unique months per user to get their activity span
    features['num_active_months'] = grp['yyyymm'].nunique()
    
    # Calculate average number of active days per month
    def calc_active_days_per_month(df: pd.DataFrame) -> float:
        # Group by month and count unique days in each month
        days_per_month = df.groupby('yyyymm')['day'].nunique()
        # Return mean number of active days across months
        return days_per_month.mean() if len(days_per_month) > 0 else 0.0
    
    features['active_days_per_month_mean'] = grp.apply(calc_active_days_per_month)

    #################################################################################################

    # Final cleanup: ensure numeric types
    features = features.fillna(0.0)
    features = features.reset_index().rename(columns={'index': 'user_id'})
    return features


def build_label_frame(transactions: pd.DataFrame) -> pd.DataFrame:
    # One label per user
    labels = transactions[['user_id','gender']].drop_duplicates('user_id').copy()
    return labels


#################################################################################################
# Model evaluation
#################################################################################################   

def eval_model(name, model, Xtr, Xva, Xtr_scaled, Xva_scaled, y_train, y_val, scaled=False):
    if scaled:
        # train, val, test
        Xtr_i, Xva_i = Xtr_scaled, Xva_scaled
    else:
        Xtr_i, Xva_i = Xtr, Xva
    yhat_tr = model.predict(Xtr_i)
    yhat_va = model.predict(Xva_i)
    metrics = {
        'model': name,
        'train_accuracy': accuracy_score(y_train, yhat_tr),
        'train_precision': precision_score(y_train, yhat_tr),
        'train_recall': recall_score(y_train, yhat_tr),
        'train_f1': f1_score(y_train, yhat_tr),
        'val_accuracy': accuracy_score(y_val, yhat_va),
        'val_precision': precision_score(y_val, yhat_va),
        'val_recall': recall_score(y_val, yhat_va),
        'val_f1': f1_score(y_val, yhat_va),
    }
    return metrics, yhat_va, yhat_tr

def test_results(name, model, Xte, Xte_scaled, y_test, scaled=False):
    if scaled:
        Xte_i = Xte_scaled
    else:
        Xte_i = Xte
    yhat_te = model.predict(Xte_i)
    metrics = {
        'model': name,
        'test_accuracy': accuracy_score(y_test, yhat_te),
        'test_precision': precision_score(y_test, yhat_te),
        'test_recall': recall_score(y_test, yhat_te),
        'test_f1': f1_score(y_test, yhat_te),
    }
    return metrics, yhat_te

