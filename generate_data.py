import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'customer_id': range(1, n + 1),
    'age': np.random.randint(18, 70, n),
    'tenure_months': np.random.randint(1, 60, n),
    'monthly_charges': np.round(np.random.uniform(20, 150, n), 2),
    'num_support_tickets': np.random.randint(0, 10, n),
    'num_products': np.random.randint(1, 5, n),
    'contract_type': np.random.choice(['monthly', 'yearly', 'two_year'], n, p=[0.5, 0.3, 0.2]),
    'payment_method': np.random.choice(['credit_card', 'bank_transfer', 'e_wallet'], n),
    'last_login_days_ago': np.random.randint(1, 180, n),
})

# Churn logic — customers churn more if high charges, many tickets, low tenure
churn_score = (
    (df['monthly_charges'] > 100).astype(int) * 2 +
    (df['num_support_tickets'] > 5).astype(int) * 2 +
    (df['tenure_months'] < 6).astype(int) * 2 +
    (df['last_login_days_ago'] > 90).astype(int) * 2 +
    (df['contract_type'] == 'monthly').astype(int) +
    np.random.randint(0, 3, n)
)

df['churned'] = (churn_score >= 5).astype(int)

df.to_csv('customer_data.csv', index=False)
print(f"Dataset created! Total customers: {n}")
print(f"Churned: {df['churned'].sum()} ({df['churned'].mean()*100:.1f}%)")
print(f"Retained: {(df['churned'] == 0).sum()} ({(1 - df['churned'].mean())*100:.1f}%)")