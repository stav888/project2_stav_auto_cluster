# pip install streamlit pandas scikit-learn matplotlib huggingface_hub python-dotenv
# streamlit run app2.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

from huggingface_hub import InferenceClient


# =========================================================
# COMPLETE YOURSELF FUNCTIONS
# =========================================================

def complete_yourself__wcss_by_k(X: np.ndarray, k_min: int, k_max: int) -> pd.DataFrame:
    """Calculate WCSS for k values using KMeans"""
    rows = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        rows.append({"k": k, "wcss": kmeans.inertia_})
    return pd.DataFrame(rows)


def complete_yourself__fit_kmeans_labels(X: np.ndarray, k: int) -> np.ndarray:
    """Fit KMeans and return cluster labels"""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    return kmeans.fit_predict(X)


@st.cache_resource
def complete_yourself__get_hf_client() -> InferenceClient:
    """Initialize and cache HuggingFace InferenceClient for LLaMA model"""
    hf_token = ""
    return InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        token=hf_token
    )


def complete_yourself__ask_llama_for_name_desc(cluster_summary: str) -> tuple[str, str]:
    """Generate cluster name and description using LLaMA on Hugging Face"""
    client = complete_yourself__get_hf_client()

    messages = [
        {
            "role": "system",
            "content": "You are a data scientist. Respond ONLY in format: NAME | DESCRIPTION. "
                       "Name up to 3 words, description up to 1 sentence."
        },
        {
            "role": "user",
            "content": f"Here is a cluster summary:\n{cluster_summary}"
        },
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=80,
            temperature=0.5,
        )
        text = response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"LLM error: {e}. Using fallback name.")
        return "Auto Cluster", cluster_summary[:120]

    if "|" in text:
        name, desc = text.split("|", 1)
        return name.strip(), desc.strip()[:200]

    return text[:30], text[30:230]



# =========================================================
# PROVIDED HELPERS (not the focus)
# =========================================================

def preprocess(df_features: pd.DataFrame) -> np.ndarray:
    """Preprocess data: scale numeric, encode categorical"""
    df_features = df_features.dropna(how="all").dropna(axis=1, how="all")

    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df_features.columns if c not in numeric_cols]

    for col in numeric_cols:
        df_features[col] = df_features[col].fillna(df_features[col].mean())
    for col in categorical_cols:
        df_features[col] = df_features[col].fillna("missing")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor.fit_transform(df_features)


def build_cluster_summary(df_features: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Build text summary of each cluster"""
    tmp = df_features.copy()
    tmp["_cluster"] = labels

    numeric_cols = tmp.select_dtypes(include=[np.number]).columns.drop("_cluster", errors="ignore")
    cat_cols = tmp.select_dtypes(exclude=[np.number]).columns

    out = {}
    for c in sorted(tmp["_cluster"].unique()):
        part = tmp[tmp["_cluster"] == c]
        lines = [f"Cluster {c}: n={len(part)}"]

        if len(numeric_cols) > 0:
            means = part[numeric_cols].mean(numeric_only=True).sort_values(ascending=False).head(8)
            means_text = ", ".join([f"{col}={means[col]:.2f}" for col in means.index])
            lines.append(f"Numeric averages (top): {means_text}")

        if len(cat_cols) > 0:
            bits = []
            for col in list(cat_cols)[:6]:
                vc = part[col].astype(str).value_counts(dropna=False).head(3)
                top_vals = "; ".join([f"{idx}({cnt})" for idx, cnt in vc.items()])
                bits.append(f"{col}: {top_vals}")
            lines.append("Categorical top values: " + " | ".join(bits))

        out[int(c)] = "\n".join(lines)

    return out


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Segment Studio", layout="wide")
st.title("Segment Studio")

st.session_state.setdefault("wcss_df", None)
st.session_state.setdefault("labels", None)
st.session_state.setdefault("cluster_counts_table", None)
st.session_state.setdefault("cluster_labels_table", None)
st.session_state.setdefault("cluster_name_map", None)

# Step 1: Upload
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to continue")
    st.stop()

df = pd.read_csv(uploaded, na_values=['?'])
st.subheader("Step 1: CSV Table")
st.dataframe(df, use_container_width=True)

df_features = df.dropna(how="all").copy()
if df_features.shape[0] < 3:
    st.error("Need at least 3 rows")
    st.stop()

X = preprocess(df_features)


# Step 2: WCSS elbow
st.subheader("Step 2: WCSS (Elbow)")

n_samples = X.shape[0]
max_allowed_k = min(20, n_samples - 1)

col1, col2 = st.columns(2)
with col1:
    k_min = st.slider("Min k", 2, max_allowed_k, 2, 1)
with col2:
    k_max = st.slider("Max k", 2, max_allowed_k, min(10, max_allowed_k), 1)

if k_min >= k_max:
    st.warning("Min k must be smaller than Max k")
    st.stop()

if st.button("Run WCSS"):
    st.session_state["wcss_df"] = complete_yourself__wcss_by_k(X, k_min, k_max)

if st.session_state["wcss_df"] is None:
    st.info("Click 'Run WCSS' to see the elbow plot")
    st.stop()

wcss_df = st.session_state["wcss_df"]
st.dataframe(wcss_df, use_container_width=True)

fig = plt.figure()
plt.plot(wcss_df["k"], wcss_df["wcss"], marker="o")
plt.xlabel("k")
plt.ylabel("WCSS (inertia)")
plt.title("Elbow Plot")
st.pyplot(fig, use_container_width=True)


# Step 3: Cluster + show counts with empty name/desc
st.subheader("Step 3: Choose k and create clusters")

chosen_k = st.slider("Select k", k_min, k_max, k_min, 1)

if st.button("Create clusters"):
    labels = complete_yourself__fit_kmeans_labels(X, chosen_k)

    st.session_state["labels"] = labels
    st.session_state["cluster_labels_table"] = None
    st.session_state["cluster_name_map"] = None

    counts = pd.Series(labels).value_counts().sort_index()
    counts_table = pd.DataFrame({
        "cluster_id": counts.index.astype(int),
        "count": counts.values.astype(int),
        "name": [""] * len(counts),
        "description": [""] * len(counts),
    })

    st.session_state["cluster_counts_table"] = counts_table

if st.session_state["cluster_counts_table"] is not None:
    st.write("Cluster counts (name/description empty for now)")
    st.dataframe(st.session_state["cluster_counts_table"], use_container_width=True)
else:
    st.info("Click 'Create clusters' to generate the clusters")
    st.stop()


# Step 4: Button to call LLM and fill name/description
st.subheader("Step 4: Generate group name + description (LLaMA)")

if st.button("Generate names/descriptions with LLaMA"):
    labels = st.session_state["labels"]
    counts_table = st.session_state["cluster_counts_table"]

    summaries = build_cluster_summary(df_features, labels)

    rows = []
    name_map = {}

    with st.spinner("Calling LLaMA for each cluster..."):
        for cluster_id in counts_table["cluster_id"].tolist():
            summary = summaries[int(cluster_id)]
            name, desc = complete_yourself__ask_llama_for_name_desc(summary)

            if not name:
                name = f"cluster_{int(cluster_id)}"
            if desc is None:
                desc = ""

            name_map[int(cluster_id)] = name
            rows.append({
                "cluster_id": int(cluster_id),
                "count": int(counts_table.loc[counts_table["cluster_id"] == cluster_id, "count"].iloc[0]),
                "name": name,
                "description": desc,
            })

    labeled_table = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    st.session_state["cluster_labels_table"] = labeled_table
    st.session_state["cluster_name_map"] = name_map

if st.session_state["cluster_labels_table"] is not None:
    st.write("Cluster labels (with name + description)")
    st.dataframe(st.session_state["cluster_labels_table"], use_container_width=True)
else:
    st.info("Click 'Generate names/descriptions with LLaMA' to fill Step 4")
    st.stop()

# Step 5: Export
st.subheader("Step 5: Export clustered CSV")

labels = st.session_state["labels"]
name_map = st.session_state["cluster_name_map"] or {}
cluster_names = [name_map.get(int(c), f"cluster_{int(c)}") for c in labels]

df_out = df.copy()
df_out["cluster_name"] = cluster_names

original_name = getattr(uploaded, "name", "input.csv")
out_name = original_name[:-4] + "_clustered.csv" if original_name.lower().endswith(".csv") else original_name + "_clustered.csv"

st.download_button(
    "Download clustered CSV",
    data=df_out.to_csv(index=False).encode("utf-8"),
    file_name=out_name,
    mime="text/csv",
)


