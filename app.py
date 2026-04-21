import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import shap

st.set_page_config(
    page_title="Churn Analysis Dashboard",
    page_icon="📉",
    layout="wide"
)

# ── Auto-detect churn column ───────────────────────────────────────────────────
def guess_churn_col(df):
    for col in df.columns:
        if any(k in col.lower() for k in ['churn', 'left', 'cancel', 'attrition', 'exit']):
            return col
    # fallback: any binary 0/1 column
    for col in df.columns:
        vals = df[col].dropna().unique()
        if set(vals).issubset({0, 1, '0', '1', True, False}):
            return col
    return None

# ── Core analysis — fully generalised ─────────────────────────────────────────
def run_analysis(df, churn_col):
    d = df.copy()

    # Normalise churn column to 0/1 int
    d[churn_col] = pd.to_numeric(d[churn_col], errors='coerce')
    d = d.dropna(subset=[churn_col])
    d[churn_col] = d[churn_col].astype(int)

    # Separate features from target
    X_raw = d.drop(columns=[churn_col])

    # Drop columns that are clearly IDs (unique ratio > 95%)
    id_cols = [
        c for c in X_raw.columns
        if X_raw[c].nunique() / max(len(X_raw), 1) > 0.95
        and X_raw[c].dtype in ['int64', 'float64', 'object']
    ]
    X_raw = X_raw.drop(columns=id_cols)

    # Split into numeric and categorical
    num_cols = X_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_raw.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # One-hot encode categoricals (max 20 unique values to avoid explosion)
    safe_cat = [c for c in cat_cols if X_raw[c].nunique() <= 20]
    if safe_cat:
        X_encoded = pd.get_dummies(X_raw[safe_cat], drop_first=True)
    else:
        X_encoded = pd.DataFrame(index=X_raw.index)

    # Combine numeric + encoded
    X = pd.concat([X_raw[num_cols], X_encoded], axis=1)
    X = X.fillna(X.median(numeric_only=True))
    feature_cols = X.columns.tolist()

    y = d[churn_col]

    if len(X) < 50:
        st.error("Need at least 50 rows to run analysis.")
        return None
    if y.nunique() < 2:
        st.error("Churn column must have both 0 and 1 values.")
        return None

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    y_prob = model.predict_proba(X_test_s)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    # SHAP
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test_s)
    sv         = np.array(shap_vals)
    shap_churn = sv[:, :, 1] if sv.ndim == 3 else sv[1]
    shap_imp   = pd.DataFrame({
        'feature':    feature_cols,
        'importance': np.abs(shap_churn).mean(axis=0)
    }).sort_values('importance', ascending=False)

    # Score ALL rows
    X_all_s = scaler.transform(X.fillna(X.median(numeric_only=True)))
    d['churn_probability'] = model.predict_proba(X_all_s)[:, 1]
    d['churn_risk'] = pd.cut(
        d['churn_probability'],
        bins=[0, 0.4, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )

    return {
        'df':          d,
        'shap_imp':    shap_imp,
        'auc':         auc,
        'num_cols':    num_cols,
        'cat_cols':    safe_cat,
        'feature_cols': feature_cols,
        'churn_col':   churn_col,
        'orig_cols':   df.columns.tolist(),
        'id_cols':     id_cols,
    }

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("Churn Analysis Dashboard")

uploaded = st.sidebar.file_uploader(
    "Upload your dataset",
    type=['csv', 'xlsx'],
    help="Any CSV or Excel file with customer data"
)

# ── Landing ────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.title("Customer Churn Analysis")
    st.markdown("### Upload any customer dataset to get started")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Step 1**\n\nUpload any CSV or Excel file from the sidebar")
    with c2:
        st.info("**Step 2**\n\nSelect which column represents churn (0 = stayed, 1 = left)")
    with c3:
        st.info("**Step 3**\n\nGet instant predictions, SHAP explanations, and a high-risk customer list")

    st.divider()
    st.markdown("""
    #### The only requirement
    Your dataset needs **one column** that marks whether a customer churned:
    - `1` or `True` = customer left
    - `0` or `False` = customer stayed

    Everything else — column names, number of columns, data types — is handled automatically.
    No column renaming needed.
    """)
    st.stop()

# ── Load file ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_file(f):
    if f.name.endswith('.xlsx'):
        return pd.read_excel(f)
    return pd.read_csv(f)

df_raw = load_file(uploaded)
st.sidebar.success(f"Loaded {len(df_raw):,} rows · {len(df_raw.columns)} columns")

# ── Churn column picker ────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("**Which column is your churn label?**")

guessed = guess_churn_col(df_raw)
churn_options = list(df_raw.columns)
default_idx   = churn_options.index(guessed) if guessed else 0

churn_col = st.sidebar.selectbox(
    "Churn column (0 = stayed, 1 = left)",
    churn_options,
    index=default_idx
)

# Show a quick preview of what values look like
val_preview = df_raw[churn_col].value_counts().head(4).to_dict()
st.sidebar.caption(f"Values found: {val_preview}")

run_btn = st.sidebar.button("Run analysis", type="primary", use_container_width=True)

if 'results' not in st.session_state:
    st.session_state.results = None
if 'last_file' not in st.session_state:
    st.session_state.last_file = None

# Reset if new file uploaded
if uploaded.name != st.session_state.last_file:
    st.session_state.results = None
    st.session_state.last_file = uploaded.name

if run_btn:
    with st.spinner("Analysing your data — training model and computing SHAP values..."):
        st.session_state.results = run_analysis(df_raw, churn_col)

# ── Pre-run preview ────────────────────────────────────────────────────────────
if st.session_state.results is None:
    st.title("Ready to analyse")
    st.markdown(f"**{len(df_raw):,} rows · {len(df_raw.columns)} columns detected**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Numeric columns** (will be used as features)")
        num = df_raw.select_dtypes(include='number').columns.tolist()
        st.dataframe(
            pd.DataFrame({'column': num, 'sample value': [df_raw[c].dropna().iloc[0] if len(df_raw[c].dropna()) > 0 else 'N/A' for c in num]}),
            use_container_width=True, hide_index=True
        )
    with c2:
        st.markdown("**Categorical columns** (will be one-hot encoded)")
        cat = df_raw.select_dtypes(include='object').columns.tolist()
        st.dataframe(
            pd.DataFrame({'column': cat, 'unique values': [df_raw[c].nunique() for c in cat]}),
            use_container_width=True, hide_index=True
        )

    st.info(f"Churn column selected: **{churn_col}** — click **Run analysis** in the sidebar when ready.")
    st.stop()

# ── Dashboard ──────────────────────────────────────────────────────────────────
res         = st.session_state.results
df          = res['df']
shap_imp    = res['shap_imp']
auc         = res['auc']
num_cols    = res['num_cols']
churn_col   = res['churn_col']

# Sidebar filters — built dynamically from actual categorical columns
st.sidebar.divider()
st.sidebar.markdown("**Filters**")

risk_filter = st.sidebar.multiselect(
    "Churn risk level",
    ['Low', 'Medium', 'High'],
    default=['Low', 'Medium', 'High']
)

filtered = df[df['churn_risk'].isin(risk_filter)]

# Dynamic categorical filter — pick the most interesting cat column
cat_cols = res['cat_cols']
if cat_cols:
    filter_col = cat_cols[0]
    options    = df[filter_col].dropna().unique().tolist()
    selected   = st.sidebar.multiselect(
        f"Filter by {filter_col}",
        options,
        default=options
    )
    filtered = filtered[filtered[filter_col].isin(selected)]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Customer Churn Dashboard")
st.caption(
    f"Showing {len(filtered):,} of {len(df):,} customers  ·  "
    f"Model AUC: {auc:.3f}  ·  "
    f"Churn column: `{churn_col}`  ·  "
    f"{len(num_cols)} numeric features used"
)

# ── KPI Cards ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total customers", f"{len(filtered):,}")
with k2:
    churned_n = int(filtered[churn_col].sum())
    st.metric("Churned", f"{churned_n:,}")
with k3:
    rate = filtered[churn_col].mean() * 100
    st.metric("Churn rate", f"{rate:.1f}%")
with k4:
    high_n = (filtered['churn_risk'] == 'High').sum()
    st.metric("High risk", f"{high_n:,}")

st.divider()

# ── Row 1: SHAP + Risk pie ─────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Top drivers of churn")
    top = shap_imp.head(10).copy()
    top['feature'] = top['feature'].str.replace('_', ' ').str.title()
    fig = px.bar(
        top, x='importance', y='feature', orientation='h',
        color='importance', color_continuous_scale='Reds',
        labels={'importance': 'Impact score (SHAP)', 'feature': ''}
    )
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Risk distribution")
    rc = filtered['churn_risk'].value_counts().reset_index()
    rc.columns = ['Risk', 'Count']
    fig2 = px.pie(
        rc, names='Risk', values='Count',
        color='Risk',
        color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Dynamic numeric charts ─────────────────────────────────────────────
# Pick top 2 most important numeric features from SHAP to plot against churn
top_numeric = [
    f for f in shap_imp['feature'].tolist()
    if f in num_cols
][:2]

if len(top_numeric) >= 1:
    c1, c2 = st.columns(2)

    with c1:
        feat = top_numeric[0]
        st.subheader(f"{feat.replace('_',' ').title()} vs churn probability")
        sample = filtered.sample(min(400, len(filtered)))
        fig3 = px.scatter(
            sample, x=feat, y='churn_probability',
            color='churn_risk',
            color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
            opacity=0.6,
            labels={
                feat: feat.replace('_', ' ').title(),
                'churn_probability': 'Churn probability',
                'churn_risk': 'Risk'
            }
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        if len(top_numeric) >= 2:
            feat2 = top_numeric[1]
            st.subheader(f"Churn rate by {feat2.replace('_',' ').title()}")
            # Bin the feature into 5 groups automatically
            try:
                filtered2 = filtered.copy()
                filtered2['_bin'] = pd.qcut(
                    filtered2[feat2], q=5, duplicates='drop'
                ).astype(str)
                binned = (
                    filtered2.groupby('_bin')[churn_col]
                    .mean() * 100
                ).reset_index()
                binned.columns = ['Group', 'Churn rate (%)']
                fig4 = px.bar(
                    binned, x='Group', y='Churn rate (%)',
                    color='Churn rate (%)',
                    color_continuous_scale='Oranges',
                    labels={'Group': feat2.replace('_', ' ').title()}
                )
                fig4.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig4, use_container_width=True)
            except Exception:
                st.info(f"Could not bin {feat2} — too few unique values.")

# ── Row 3: Categorical breakdown (if any cat cols exist) ───────────────────────
if cat_cols:
    st.subheader("Churn rate by category")
    cols_to_show = cat_cols[:3]  # show up to 3 categorical breakdowns
    chart_cols   = st.columns(len(cols_to_show))

    for i, col in enumerate(cols_to_show):
        with chart_cols[i]:
            breakdown = (
                filtered.groupby(col)[churn_col]
                .mean() * 100
            ).reset_index()
            breakdown.columns = [col, 'Churn rate (%)']
            breakdown = breakdown.sort_values('Churn rate (%)', ascending=False)
            fig5 = px.bar(
                breakdown, x=col, y='Churn rate (%)',
                color='Churn rate (%)',
                color_continuous_scale='RdYlGn_r',
                title=col.replace('_', ' ').title()
            )
            fig5.update_layout(
                coloraxis_showscale=False,
                title_font_size=14,
                margin=dict(t=40)
            )
            st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ── High risk table ────────────────────────────────────────────────────────────
st.subheader("High risk customers — action needed")

high_risk = filtered[filtered['churn_risk'] == 'High'].copy()
high_risk = high_risk.sort_values('churn_probability', ascending=False)

# Show original columns + churn probability
orig_cols    = [c for c in res['orig_cols'] if c in high_risk.columns]
display_cols = orig_cols + ['churn_probability', 'churn_risk']
display_cols = list(dict.fromkeys(display_cols))

show_df = high_risk[display_cols].head(50).copy()
show_df['churn_probability'] = (
    show_df['churn_probability'] * 100
).round(1).astype(str) + '%'

st.dataframe(show_df, use_container_width=True, hide_index=True)

csv = high_risk[display_cols].to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download high risk list as CSV",
    data=csv,
    file_name='high_risk_customers.csv',
    mime='text/csv'
)

# ── Data summary expander ──────────────────────────────────────────────────────
with st.expander("View full dataset summary"):
    st.dataframe(df_raw.describe(include='all').T, use_container_width=True)