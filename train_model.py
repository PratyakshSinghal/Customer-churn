import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score)
import shap

# --- Load prepared data ---
X_train, X_test, y_train, y_test, features, scaler = pickle.load(
    open('prepared_data.pkl', 'rb')
)

# --- Train the model ---
print("Training model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)
print("Done!")

# --- Evaluate ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Model Performance ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.3f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

# --- SHAP explainability ---
print("\nCalculating SHAP values (why customers churn)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification, take churn class (index 1)
shap_churn = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values[1]

# Average importance per feature
shap_importance = pd.DataFrame({
    'feature': features,
    'importance': np.abs(shap_churn).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\n=== Top reasons customers churn (SHAP) ===")
print(shap_importance.to_string(index=False))

# --- Save model + explainer ---
pickle.dump({
    'model': model,
    'explainer': explainer,
    'features': features,
    'scaler': scaler,
    'shap_importance': shap_importance,
    'X_test': X_test,
    'y_test': y_test,
    'y_pred': y_pred,
    'y_prob': y_prob,
    'shap_churn': shap_churn
}, open('model.pkl', 'wb'))

print("\nmodel.pkl saved. Ready for dashboard!")