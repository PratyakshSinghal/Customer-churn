import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('customer_data.csv')

# --- Feature engineering ---
# Higher charges relative to tenure = more likely to churn
df['charge_per_month_tenure'] = df['monthly_charges'] / (df['tenure_months'] + 1)

# Customers inactive for long time are at risk
df['is_inactive'] = (df['last_login_days_ago'] > 60).astype(int)

# High support tickets = frustrated customer
df['is_high_support'] = (df['num_support_tickets'] >= 5).astype(int)

# --- Convert text columns to numbers ---
df['contract_monthly'] = (df['contract_type'] == 'monthly').astype(int)
df['contract_yearly'] = (df['contract_type'] == 'yearly').astype(int)

df['pay_credit'] = (df['payment_method'] == 'credit_card').astype(int)
df['pay_bank'] = (df['payment_method'] == 'bank_transfer').astype(int)

# --- Select final features ---
features = [
    'age',
    'tenure_months',
    'monthly_charges',
    'num_support_tickets',
    'num_products',
    'last_login_days_ago',
    'charge_per_month_tenure',
    'is_inactive',
    'is_high_support',
    'contract_monthly',
    'contract_yearly',
    'pay_credit',
    'pay_bank',
]

X = df[features]
y = df['churned']

# --- Split into train and test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Scale the numbers ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Save everything for next phase ---
pickle.dump((X_train_scaled, X_test_scaled, y_train, y_test, features, scaler), 
            open('prepared_data.pkl', 'wb'))

print("Features used:", features)
print(f"\nTraining set size: {X_train.shape[0]} customers")
print(f"Test set size:     {X_test.shape[0]} customers")
print("\nFeature engineering done. prepared_data.pkl saved.")