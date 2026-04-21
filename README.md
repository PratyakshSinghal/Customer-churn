# Customer Churn Analysis Dashboard

A machine learning powered dashboard that helps businesses understand **why customers are leaving** — and identify who is most at risk before they go.

Built with Python, scikit-learn, SHAP, and Streamlit.

---

## Live Demo

> Upload any customer CSV → map your columns → get instant churn predictions, explainability charts, and a downloadable high-risk customer list.

---

## Features

- **Upload any dataset** — works with any CSV or Excel file with customer data
- **Auto column mapping** — smart detection of your column names
- **ML churn prediction** — Random Forest model trained on your data in real time
- **SHAP explainability** — tells you *why* customers are churning, not just who
- **Interactive filters** — filter by risk level, contract type, tenure
- **High risk customer table** — exportable list of customers who need immediate attention
- **Key charts included:**
  - Top churn reasons (SHAP importance)
  - Churn risk distribution (Low / Medium / High)
  - Monthly charges vs churn probability
  - Churn rate by tenure group
  - Support tickets vs churn rate

---

## Tech Stack

| Layer | Technology |
|---|---|
| Dashboard UI | Streamlit |
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn (Random Forest) |
| Explainability | SHAP |
| Charts | Plotly Express |
| File handling | openpyxl (Excel support) |
| Language | Python 3.10+ |

---

## Project Structure

```
churn-dashboard/
│
├── app.py                  # Main Streamlit dashboard (upload-based)
├── generate_data.py        # Generates dummy customer dataset for testing
├── explore_data.py         # Exploratory data analysis script
├── prepare_data.py         # Feature engineering and data prep
├── train_model.py          # Model training and SHAP analysis
│
├── customer_data.csv       # Sample dummy dataset (1000 customers)
├── prepared_data.pkl       # Preprocessed features (generated)
├── model.pkl               # Trained model bundle (generated)
│
├── requirements.txt        # All dependencies
└── README.md               # This file
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/churn-dashboard.git
cd churn-dashboard
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Using the Dashboard

### Option A — Use the included sample data
Upload `customer_data.csv` from this repo. The column mapping will auto-detect everything.

### Option B — Use your own data
Your CSV should ideally contain columns like:

| Column type | Examples |
|---|---|
| Churn label (required) | churned, churn, left, cancelled |
| Tenure / months active | tenure_months, months_active, duration |
| Monthly charges | monthly_charges, spend, revenue, amount |
| Support tickets | num_tickets, complaints, support_count |
| Last activity | last_login_days_ago, days_inactive |
| Contract type | contract_type, plan, subscription |

Don't worry if your column names are different — the dashboard lets you map them manually.

---

## Sample Output

### KPI Cards
Shows total customers, number churned, churn rate %, and high risk count at a glance.

### Top Reasons for Churn (SHAP)
A horizontal bar chart ranked by how much each feature drives churn. The longer the bar, the bigger the driver. This is the most actionable insight — fix the top 2–3 factors and churn will drop.

### High Risk Customer List
A sortable table of your highest-risk customers with a one-click CSV download — ready to hand to your customer success team.

---

## Generate Sample Data

To regenerate or customise the dummy dataset:

```bash
python generate_data.py
```

This creates `customer_data.csv` with 1000 synthetic customers and realistic churn patterns based on monthly charges, support tickets, tenure, inactivity, and contract type.

---

## How the Model Works

1. Raw data is loaded and cleaned
2. New features are engineered (e.g. charge per tenure month, inactivity flag)
3. Text columns are one-hot encoded automatically
4. A Random Forest classifier is trained on 80% of the data
5. The remaining 20% is used to evaluate accuracy and AUC score
6. SHAP TreeExplainer calculates feature contributions for every prediction
7. All customers are scored with a churn probability (0–1)
8. Customers are grouped into Low / Medium / High risk bands

---

## Requirements

```
pandas
numpy
scikit-learn
shap
streamlit
plotly
openpyxl
```

---

## Roadmap / Future Ideas

- [ ] Connect to live database (PostgreSQL / MySQL)
- [ ] Add email alerts for high risk customers
- [ ] Time series view of churn trends over months
- [ ] Customer segmentation clustering (KMeans)
- [ ] Deploy to Streamlit Cloud for public access

---

## Author

Built by Pratyaksh Singhal


MIT License — free to use, modify, and share.
