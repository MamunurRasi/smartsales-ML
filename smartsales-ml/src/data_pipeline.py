# ============================================================
# SmartSales ML - Customer Segmentation using RFM + K-Means
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(file_path):
    """
    Load retail dataset.
    """
    df = pd.read_csv(file_path, encoding="latin1")
    return df


# ============================================================
# 2. DATA CLEANING
# ============================================================

def clean_data(df):
    """
    Clean retail dataset.
    """

    # Remove missing customer IDs
    df = df.dropna(subset=["Customer ID"])

    # Remove invalid prices
    df = df[df["Price"] > 0]

    # Remove cancelled invoices
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # Convert datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Revenue column
    df["Revenue"] = df["Quantity"] * df["Price"]

    return df


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

def perform_eda(df):

    print("\n==============================")
    print("DATASET OVERVIEW")
    print("==============================")

    print(df.info())
    print(df.describe())

    # Total Revenue
    total_revenue = df["Revenue"].sum()
    print(f"\nTotal Revenue: {total_revenue:,.2f}")

    # Revenue by Country
    country_revenue = (
        df.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nTop 10 Countries by Revenue:")
    print(country_revenue.head(10))

    plt.figure(figsize=(10, 5))
    country_revenue.head(10).plot(kind="bar")
    plt.title("Top 10 Countries by Revenue")
    plt.ylabel("Revenue")
    plt.show()

    # Best Selling Products
    top_products = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nTop 10 Products:")
    print(top_products.head(10))

    plt.figure(figsize=(10, 5))
    top_products.head(10).plot(kind="bar")
    plt.title("Top 10 Best Selling Products")
    plt.ylabel("Quantity Sold")
    plt.show()

    # Daily Sales Trend
    df["Date"] = df["InvoiceDate"].dt.date

    daily_sales = (
        df.groupby("Date")["Revenue"]
        .sum()
    )

    plt.figure(figsize=(12, 5))
    daily_sales.plot()
    plt.title("Daily Sales Trend")
    plt.ylabel("Revenue")
    plt.show()

    # Top Customers
    top_customers = (
        df.groupby("Customer ID")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nTop 10 Customers:")
    print(top_customers.head(10))

    plt.figure(figsize=(10, 5))
    top_customers.head(10).plot(kind="bar")
    plt.title("Top 10 Customers by Revenue")
    plt.ylabel("Revenue")
    plt.show()


# ============================================================
# 4. RFM FEATURE ENGINEERING
# ============================================================

def create_rfm(df):

    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,
        "Invoice": "nunique",
        "Revenue": "sum"
    })

    rfm.columns = [
        "Recency",
        "Frequency",
        "Monetary"
    ]

    rfm = rfm.reset_index()

    return rfm


# ============================================================
# 5. CUSTOMER SEGMENTATION
# ============================================================

def assign_rfm_scores(rfm):

    rfm["R_score"] = pd.qcut(
        rfm["Recency"],
        5,
        labels=[5, 4, 3, 2, 1]
    )

    rfm["F_score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    )

    rfm["M_score"] = pd.qcut(
        rfm["Monetary"],
        5,
        labels=[1, 2, 3, 4, 5]
    )

    rfm["RFM_Score"] = (
        rfm["R_score"].astype(str)
        + rfm["F_score"].astype(str)
    )

    return rfm


def customer_segment(row):

    if row["RFM_Score"] == "55":
        return "Champions"

    elif row["RFM_Score"][1] == "5":
        return "Loyal Customers"

    elif row["RFM_Score"][0] == "5":
        return "New Customers"

    elif row["RFM_Score"][0] == "1":
        return "At Risk"

    else:
        return "Regular"


def create_segments(rfm):

    rfm["Segment"] = rfm.apply(
        customer_segment,
        axis=1
    )

    return rfm


# ============================================================
# 6. K-MEANS CLUSTERING
# ============================================================

def perform_clustering(rfm):

    X = rfm[
        ["Recency", "Frequency", "Monetary"]
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method
    inertia = []

    for k in range(1, 11):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)

        inertia.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertia, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Clusters")
    plt.ylabel("Inertia")
    plt.show()

    # Final Model
    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = kmeans.fit_predict(
        X_scaled
    )

    print("\nCluster Summary")
    print(
        rfm.groupby("Cluster")[
            ["Recency", "Frequency", "Monetary"]
        ].mean()
    )

    return rfm, X_scaled, kmeans


# ============================================================
# 7. PCA VISUALIZATION
# ============================================================

def visualize_clusters(rfm, X_scaled):

    pca = PCA(n_components=2)

    components = pca.fit_transform(X_scaled)

    rfm["PCA1"] = components[:, 0]
    rfm["PCA2"] = components[:, 1]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        rfm["PCA1"],
        rfm["PCA2"],
        c=rfm["Cluster"],
        cmap="viridis"
    )

    plt.title("Customer Clusters")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    plt.show()


# ============================================================
# 8. MAIN PIPELINE
# ============================================================

def main():

    FILE_PATH = "data/raw/online_retail_II_2009_2010.csv"

    # Load
    df = load_data(FILE_PATH)

    # Clean
    df = clean_data(df)

    # EDA
    perform_eda(df)

    # RFM
    rfm = create_rfm(df)

    # Segmentation
    rfm = assign_rfm_scores(rfm)
    rfm = create_segments(rfm)

    print("\nCustomer Segments")
    print(rfm["Segment"].value_counts())

    # Clustering
    rfm, X_scaled, model = perform_clustering(rfm)

    # Visualization
    visualize_clusters(rfm, X_scaled)

    return rfm


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    rfm_data = main()