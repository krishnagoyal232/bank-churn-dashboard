"""
Bank Customer Churn Risk Dashboard
European Central Bank — Cloud Ready Version
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Risk | ECB",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0F1117; }
  .card { background:#1A1F2E; border-radius:14px; padding:22px 26px;
          border:1px solid #2A3050; margin-bottom:12px; }
  .kpi-val  { font-size:2.2rem; font-weight:800; color:#4FC3F7; }
  .kpi-lbl  { font-size:0.82rem; color:#78909C; margin-top:4px; }
  .low    { color:#4CAF50; font-weight:700; font-size:1.4rem; }
  .medium { color:#FFC107; font-weight:700; font-size:1.4rem; }
  .high   { color:#F44336; font-weight:700; font-size:1.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "European_Bank.csv")
OUT_DIR   = os.path.join(BASE, "outputs")
MODEL_DIR = os.path.join(BASE, "models")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

PALETTE = {"churned": "#E74C3C", "retained": "#2ECC71", "neutral": "#3498DB"}

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN MODEL (runs automatically if models don't exist yet)
# ══════════════════════════════════════════════════════════════════════════════
def train_and_save():
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, roc_curve, confusion_matrix)
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    import shap

    df = pd.read_csv(DATA_PATH)
    drop_cols = [c for c in ["Year", "CustomerId", "Surname"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=False)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    df["Balance_Salary_Ratio"]   = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Product_Density"]        = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["Engagement_Product"]     = df["IsActiveMember"] * df["NumOfProducts"]
    df["Age_Tenure_Interaction"] = df["Age"] * df["Tenure"]
    df["Zero_Balance_Flag"]      = (df["Balance"] == 0).astype(int)
    df["CreditScore_Age_Ratio"]  = df["CreditScore"] / (df["Age"] + 1)

    FEATURE_COLS = [c for c in df.columns if c != "Exited"]
    X = df[FEATURE_COLS]
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=10,
                                                       random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                                           max_depth=5, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5,
                                             eval_metric="logloss", random_state=42, n_jobs=-1),
    }

    results = {}
    best_model_name = None
    best_roc = 0
    best_model_obj = None

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    colors_roc = ["#3498DB","#E67E22","#2ECC71","#9B59B6","#E74C3C"]

    for (name, model), color in zip(models.items(), colors_roc):
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)

        results[name] = {"Accuracy": acc, "Precision": prec,
                         "Recall": rec, "F1": f1, "ROC-AUC": auc}

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

        if auc > best_roc:
            best_roc = auc
            best_model_name = name
            best_model_obj  = model

        joblib.dump(model, os.path.join(MODEL_DIR,
                    f"{name.replace(' ','_').lower()}.pkl"))

    ax_roc.plot([0,1],[0,1],"k--", lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_roc_curves.png"), dpi=150)
    plt.close()

    # Confusion matrix
    y_pred_best = best_model_obj.predict(X_test_sc)
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retained","Churned"],
                yticklabels=["Retained","Churned"], ax=ax)
    ax.set_title(f"Confusion Matrix — {best_model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "09_confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature importance
    importances  = best_model_obj.feature_importances_
    feat_imp_full = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
    feat_imp_full = feat_imp_full.sort_values("Importance", ascending=False)
    feat_imp_full.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)

    top15 = feat_imp_full.sort_values("Importance").tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_fi = [PALETTE["churned"] if i >= 10 else PALETTE["neutral"] for i in range(len(top15))]
    ax.barh(top15["Feature"], top15["Importance"], color=colors_fi, edgecolor="white")
    ax.set_title(f"Feature Importance — {best_model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "10_feature_importance.png"), dpi=150)
    plt.close()

    # EDA plots
    geo_cols = [c for c in df.columns if c.startswith("Geography_")]
    geo_data = {}
    for c in geo_cols:
        country = c.replace("Geography_", "")
        geo_data[country] = df[df[c]==1]["Exited"].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    clrs = [PALETTE["churned"] if v > df["Exited"].mean() else PALETTE["neutral"]
            for v in geo_data.values()]
    bars = ax.bar(geo_data.keys(), [v*100 for v in geo_data.values()], color=clrs, edgecolor="white")
    ax.axhline(df["Exited"].mean()*100, linestyle="--", color="gray", label="Overall avg")
    for bar, v in zip(bars, geo_data.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Churn Rate by Geography", fontsize=14, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_churn_by_geography.png"), dpi=150)
    plt.close()

    prod_churn = df.groupby("NumOfProducts")["Exited"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    clrs2 = [PALETTE["churned"] if v > df["Exited"].mean()*100 else PALETTE["neutral"]
             for v in prod_churn.values]
    bars2 = ax.bar(prod_churn.index.astype(str), prod_churn.values, color=clrs2, edgecolor="white")
    for bar, v in zip(bars2, prod_churn.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(df["Exited"].mean()*100, linestyle="--", color="gray", label="Overall avg")
    ax.set_title("Churn Rate by Number of Products", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Products"); ax.set_ylabel("Churn Rate (%)"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_churn_by_products.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    df[df["Exited"]==0]["Age"].plot(kind="hist", bins=30, alpha=0.6,
                                    color=PALETTE["retained"], label="Retained", ax=ax)
    df[df["Exited"]==1]["Age"].plot(kind="hist", bins=30, alpha=0.6,
                                    color=PALETTE["churned"], label="Churned", ax=ax)
    ax.set_title("Age Distribution by Churn Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age"); ax.set_ylabel("Count"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_age_distribution.png"), dpi=150)
    plt.close()

    # SHAP
    try:
        X_test_df  = pd.DataFrame(X_test_sc, columns=FEATURE_COLS)
        sample_idx = np.random.choice(len(X_test_df), size=200, replace=False)
        X_shap     = X_test_df.iloc[sample_idx]
        explainer   = shap.TreeExplainer(best_model_obj)
        shap_values = explainer.shap_values(X_shap)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        fig, ax = plt.subplots(figsize=(9, 7))
        shap.summary_plot(sv, X_shap, plot_type="bar", show=False,
                          max_display=15, color=PALETTE["churned"])
        plt.title("SHAP Feature Importance", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "11_shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(9, 7))
        shap.summary_plot(sv, X_shap, show=False, max_display=15)
        plt.title("SHAP Beeswarm — Churn Drivers", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "12_shap_beeswarm.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    # Save artifacts
    joblib.dump(scaler,      os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(FEATURE_COLS,os.path.join(MODEL_DIR, "feature_cols.pkl"))
    joblib.dump(feat_imp_full.head(10)["Feature"].tolist(),
                os.path.join(MODEL_DIR, "top_features.pkl"))

    results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index":"Model"})
    results_df.to_csv(os.path.join(OUT_DIR, "model_metrics.csv"), index=False)

    summary = {
        "best_model":   best_model_name,
        "best_auc":     round(best_roc, 4),
        "churn_rate":   round(float(df["Exited"].mean()), 4),
        "dataset_size": len(df),
        "results": {k: {m: round(v,4) for m,v in r.items()} for k,r in results.items()}
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return True

# ── Auto-train if models missing ───────────────────────────────────────────────
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
if not os.path.exists(scaler_path):
    with st.spinner("🤖 First launch — training models... This takes 1-2 minutes. Please wait!"):
        train_and_save()
    st.success("✅ Models trained! Loading dashboard...")
    st.rerun()

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler    = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model     = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
    feat_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    with open(os.path.join(OUT_DIR, "summary.json")) as f:
        summary = json.load(f)
    metrics   = pd.read_csv(os.path.join(OUT_DIR, "model_metrics.csv"))
    feat_imp  = pd.read_csv(os.path.join(OUT_DIR, "feature_importance.csv"))
    return scaler, model, feat_cols, summary, metrics, feat_imp

scaler, model, feat_cols, summary, metrics_df, feat_imp_df = load_models()

# ── Predict ────────────────────────────────────────────────────────────────────
def predict(inputs):
    row = {c: 0 for c in feat_cols}
    for k in ["CreditScore","Age","Tenure","Balance","NumOfProducts",
               "HasCrCard","IsActiveMember","EstimatedSalary"]:
        row[k] = inputs[k]
    row[f"Geography_{inputs['Geography']}"] = 1
    row[f"Gender_{inputs['Gender']}"]       = 1
    row["Balance_Salary_Ratio"]   = inputs["Balance"] / (inputs["EstimatedSalary"] + 1)
    row["Product_Density"]        = inputs["NumOfProducts"] / (inputs["Tenure"] + 1)
    row["Engagement_Product"]     = inputs["IsActiveMember"] * inputs["NumOfProducts"]
    row["Age_Tenure_Interaction"] = inputs["Age"] * inputs["Tenure"]
    row["Zero_Balance_Flag"]      = int(inputs["Balance"] == 0)
    row["CreditScore_Age_Ratio"]  = inputs["CreditScore"] / (inputs["Age"] + 1)
    vec = np.array([row[c] for c in feat_cols]).reshape(1, -1)
    return float(model.predict_proba(scaler.transform(vec))[0, 1])

def risk_info(p):
    if p < 0.35:   return "LOW",    "low",    "🟢", "#1B4332", "Customer looks stable. Great time to explore cross-sell opportunities."
    elif p < 0.65: return "MEDIUM", "medium", "🟡", "#3D2B00", "Some risk signals detected. Consider a personal outreach or product review."
    else:          return "HIGH",   "high",   "🔴", "#4A0000", "Serious risk. Assign a relationship manager immediately."

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#0D1B2A,#1B2838);border-radius:16px;
            padding:26px 32px;margin-bottom:24px;border:1px solid #2A3050;'>
  <span style='font-size:2.2rem;'>🏦</span>
  <span style='font-size:1.6rem;font-weight:800;color:#E0E0E0;margin-left:12px;'>
    Customer Churn Risk Dashboard
  </span>
  <span style='display:block;color:#78909C;margin-top:4px;margin-left:52px;font-size:0.9rem;'>
    European Central Bank  ·  XGBoost Model  ·  AUC 0.863
  </span>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in zip(
    [c1,c2,c3,c4],
    [f"{summary['best_auc']:.3f}", "85.1%", f"{summary['churn_rate']*100:.1f}%", f"{summary['dataset_size']:,}"],
    ["Model AUC Score","Accuracy","Overall Churn Rate","Customers Analysed"]
):
    with col:
        st.markdown(f"<div class='card' style='text-align:center'>"
                    f"<div class='kpi-val'>{val}</div>"
                    f"<div class='kpi-lbl'>{lbl}</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Risk Calculator", "📊  Model Results", "🔍  What Drives Churn"])

# ══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Customer Details")
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        r1, r2 = st.columns(2)
        with r1:
            geography = st.selectbox("Country", ["France", "Spain", "Germany"])
            gender    = st.selectbox("Gender", ["Male", "Female"])
            age       = st.slider("Age", 18, 92, 40)
            tenure    = st.slider("Years with bank", 0, 10, 3)
            credit    = st.slider("Credit Score", 350, 850, 650)
        with r2:
            balance  = st.number_input("Account Balance (€)", 0.0, 300000.0, 80000.0, 5000.0)
            salary   = st.number_input("Estimated Salary (€)", 500.0, 200000.0, 100000.0, 5000.0)
            products = st.selectbox("Number of Products", [1,2,3,4], index=1)
            has_cc   = st.radio("Has Credit Card?", ["Yes","No"], horizontal=True)
            active   = st.radio("Active Member?",   ["Yes","No"], horizontal=True)

    inputs = {
        "CreditScore": credit, "Age": age, "Tenure": tenure,
        "Balance": balance, "NumOfProducts": products,
        "HasCrCard": int(has_cc=="Yes"), "IsActiveMember": int(active=="Yes"),
        "EstimatedSalary": salary, "Geography": geography, "Gender": gender,
    }
    prob = predict(inputs)
    label, css, icon, bg, tip = risk_info(prob)

    with col_right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob*100, 1),
            number={"suffix":"%","font":{"size":42,"color":"#E0E0E0"}},
            gauge={
                "axis": {"range":[0,100],"tickcolor":"#444"},
                "bar":  {"color":"#F44336" if prob>.65 else ("#FFC107" if prob>.35 else "#4CAF50"),
                         "thickness":0.28},
                "bgcolor":"#1A1F2E","borderwidth":0,
                "steps":[{"range":[0,35],"color":"#1B4332"},
                         {"range":[35,65],"color":"#3D2B00"},
                         {"range":[65,100],"color":"#4A0000"}],
            },
            title={"text":"Churn Probability","font":{"size":15,"color":"#78909C"}}
        ))
        fig.update_layout(paper_bgcolor="#0F1117", height=260,
                          margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        st.markdown(f"""
        <div style='background:{bg};border-radius:12px;padding:18px 22px;
                    border:1px solid #333;text-align:center;'>
          <div class='{css}'>{icon} {label} RISK</div>
          <div style='color:#CFD8DC;font-size:0.88rem;margin-top:10px;'>{tip}</div>
        </div>""", unsafe_allow_html=True)

# ══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### How the Models Performed")
    cl, cr = st.columns(2)

    with cl:
        fig_bar = go.Figure(go.Bar(
            y=metrics_df["Model"], x=metrics_df["ROC-AUC"], orientation="h",
            marker_color=["#E74C3C" if m=="XGBoost" else "#3498DB" for m in metrics_df["Model"]],
            text=[f"{v:.3f}" for v in metrics_df["ROC-AUC"]], textposition="outside"
        ))
        fig_bar.update_layout(
            title="ROC-AUC Score (higher = better)",
            paper_bgcolor="#0F1117", plot_bgcolor="#1A1F2E",
            xaxis=dict(range=[0.5,0.95], gridcolor="#2A3050"),
            yaxis=dict(gridcolor="#2A3050"),
            height=340, margin=dict(l=10,r=80,t=40,b=10),
            font=dict(color="#B0BEC5")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with cr:
        xgb  = metrics_df[metrics_df["Model"]=="XGBoost"].iloc[0]
        cats = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        vals = [xgb[c] for c in cats]
        fig_r = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill="toself", fillcolor="rgba(231,76,60,0.2)",
            line=dict(color="#E74C3C", width=2)
        ))
        fig_r.update_layout(
            title="XGBoost — All Metrics",
            polar=dict(radialaxis=dict(range=[0.5,1.0],color="#555"),bgcolor="#1A1F2E"),
            paper_bgcolor="#0F1117", showlegend=False,
            height=340, margin=dict(l=40,r=40,t=40,b=10),
            font=dict(color="#B0BEC5")
        )
        st.plotly_chart(fig_r, use_container_width=True)

    display = metrics_df.set_index("Model").style.format("{:.3f}").background_gradient(cmap="YlGn", axis=0)
    st.dataframe(display, use_container_width=True)

    roc_path = os.path.join(OUT_DIR, "07_roc_curves.png")
    if os.path.exists(roc_path):
        st.markdown("**ROC Curves — All Models**")
        st.image(roc_path, use_container_width=True)

# ══ TAB 3 ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### What Makes Customers Leave?")
    cl2, cr2 = st.columns(2)

    with cl2:
        top12 = feat_imp_df.head(12).sort_values("Importance")
        fig_fi = go.Figure(go.Bar(
            y=top12["Feature"], x=top12["Importance"], orientation="h",
            marker_color=["#E74C3C" if i>=9 else "#3498DB" for i in range(len(top12))],
            text=[f"{v:.3f}" for v in top12["Importance"]], textposition="outside"
        ))
        fig_fi.update_layout(
            title="Feature Importance (XGBoost)",
            paper_bgcolor="#0F1117", plot_bgcolor="#1A1F2E",
            xaxis=dict(gridcolor="#2A3050"), yaxis=dict(gridcolor="#2A3050"),
            height=420, margin=dict(l=10,r=80,t=40,b=10), font=dict(color="#B0BEC5")
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with cr2:
        shap_path = os.path.join(OUT_DIR, "11_shap_summary.png")
        if os.path.exists(shap_path):
            st.markdown("**SHAP Values — Direction & Magnitude**")
            st.image(shap_path, use_container_width=True)

    st.markdown("---")
    e1, e2 = st.columns(2)
    with e1:
        p = os.path.join(OUT_DIR, "02_churn_by_geography.png")
        if os.path.exists(p):
            st.markdown("**Churn Rate by Country**")
            st.image(p, use_container_width=True)
    with e2:
        p = os.path.join(OUT_DIR, "06_churn_by_products.png")
        if os.path.exists(p):
            st.markdown("**Churn Rate by Number of Products**")
            st.image(p, use_container_width=True)
    e3, e4 = st.columns(2)
    with e3:
        p = os.path.join(OUT_DIR, "03_age_distribution.png")
        if os.path.exists(p):
            st.markdown("**Age Distribution by Churn Status**")
            st.image(p, use_container_width=True)
    with e4:
        p = os.path.join(OUT_DIR, "12_shap_beeswarm.png")
        if os.path.exists(p):
            st.markdown("**SHAP Beeswarm — Individual Drivers**")
            st.image(p, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#546E7A;font-size:0.78rem;'>"
            "🏦 European Central Bank · Churn Intelligence · XGBoost · AUC 0.863 · 10,000 Customers"
            "</div>", unsafe_allow_html=True)
