import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

import io
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Streamlit App Config
# ---------------------------
st.set_page_config(page_title="Credit Card Usage Segmentation", layout="wide")
st.title("💳 Credit Card Customer Segmentation App")

# ---------------------------
# Sidebar - Cluster Control
# ---------------------------
st.sidebar.header("Configuration")
n_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=6, value=3)
show_dendrogram = st.sidebar.checkbox("Show Dendrogram", value=False)
show_silhouette = st.sidebar.checkbox("Evaluate Silhouette Scores", value=True)

# ---------------------------
# Upload Section
# ---------------------------
st.subheader("Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("Sample Data:")
    st.dataframe(df.head())

    if 'CUST_ID' in df.columns:
        df.drop('CUST_ID', axis=1, inplace=True)

    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Feature Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # ---------------------------
    # Show Dendrogram
    # ---------------------------
    if show_dendrogram:
        st.subheader("Dendrogram")
        plt.figure(figsize=(10, 5))
        Z = linkage(scaled_data, method='ward')
        dendrogram(Z, truncate_mode="level", p=5)
        st.pyplot(plt)

    # ---------------------------
    # Silhouette Score Evaluation
    # ---------------------------
    if show_silhouette:
        st.subheader("Silhouette Scores")
        scores = {}
        for k in range(2, 7):
            model = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = model.fit_predict(scaled_data)
            score = silhouette_score(scaled_data, labels)
            scores[k] = score
            st.write(f"n_clusters = {k} ➤ Silhouette Score = {score:.4f}")

    # ---------------------------
    # Final Agglomerative Model
    # ---------------------------
    st.subheader("Cluster Segmentation Output")
    final_model = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = final_model.fit_predict(scaled_data)

    df['CLUSTER'] = cluster_labels
    st.write("Cluster Summary Table:")
    st.dataframe(df.groupby('CLUSTER').mean().round(2))

    # ---------------------------
    # PCA Visualization
    # ---------------------------
    st.subheader("PCA-Based Cluster Visualization")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='tab10', s=50)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    st.pyplot(fig)

    # ---------------------------
    # Input Section for Prediction (Optional)
    # ---------------------------
    st.subheader("Simulate New Customer Segmentation")
    input_dict = {}
    for column in df.columns[:-1]:  # Exclude CLUSTER
        input_dict[column] = st.number_input(f"{column}", value=float(df[column].mean()), step=1.0)

    if st.button("Predict Segment"):
        input_df = pd.DataFrame([input_dict])
        scaled_input = scaler.transform(input_df)
        pred_cluster = final_model.fit_predict(np.vstack([scaled_data, scaled_input]))[-1]
        st.success(f"The new customer falls into **Cluster {pred_cluster}**.")

else:
    st.info("👆 Upload a CSV file to begin. Recommended format: Credit Card usage dataset with numerical features.")
