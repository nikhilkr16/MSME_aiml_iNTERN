import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and explore the Mall Customers dataset"""
    print("Loading Mall Customers Dataset...")
    df = pd.read_csv('Mall_Customers.csv')
    
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
    
    print("\nStatistical summary:")
    print(df.describe())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nGender distribution:")
    print(df['Gender'].value_counts())
    
    return df

def preprocess_data(df):
    """Preprocess the data for clustering"""
    print("\nPreprocessing data...")
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Encode categorical variable (Gender)
    le = LabelEncoder()
    df_processed['Gender_Encoded'] = le.fit_transform(df_processed['Gender'])
    
    # Select features for clustering
    # We'll use Age, Annual Income, and Spending Score
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df_processed[features].copy()
    
    # Also create a dataset with Gender encoded for comparison
    features_with_gender = ['Gender_Encoded', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X_with_gender = df_processed[features_with_gender].copy()
    
    print(f"Features selected: {features}")
    print(f"Feature matrix shape: {X.shape}")
    
    return X, X_with_gender, df_processed

def find_optimal_clusters(X, max_clusters=10):
    """Find optimal number of clusters using elbow method and silhouette score"""
    print(f"\nFinding optimal number of clusters (testing 1 to {max_clusters})...")
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouette_scores = []
    K_range = range(1, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        
        if k > 1:  # Silhouette score requires at least 2 clusters
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)
    
    # Plot elbow curve and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow method
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)
    
    # Silhouette scores
    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimal_clusters_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal k based on silhouette score
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
    print(f"Best silhouette score: {max(silhouette_scores):.3f}")
    
    return optimal_k, scaler

def perform_clustering(X, n_clusters, scaler):
    """Perform K-Means clustering"""
    print(f"\nPerforming K-Means clustering with {n_clusters} clusters...")
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    
    return cluster_labels, kmeans

def visualize_clusters(df, X, cluster_labels, n_clusters):
    """Visualize the clustering results"""
    print("\nVisualizing clustering results...")
    
    # Add cluster labels to dataframe
    df_viz = df.copy()
    df_viz['Cluster'] = cluster_labels
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Age vs Annual Income
    scatter1 = axes[0, 0].scatter(df_viz['Age'], df_viz['Annual Income (k$)'], 
                                 c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Annual Income (k$)')
    axes[0, 0].set_title('Clusters: Age vs Annual Income')
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # 2. Annual Income vs Spending Score
    scatter2 = axes[0, 1].scatter(df_viz['Annual Income (k$)'], df_viz['Spending Score (1-100)'], 
                                 c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[0, 1].set_xlabel('Annual Income (k$)')
    axes[0, 1].set_ylabel('Spending Score (1-100)')
    axes[0, 1].set_title('Clusters: Annual Income vs Spending Score')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # 3. Age vs Spending Score
    scatter3 = axes[1, 0].scatter(df_viz['Age'], df_viz['Spending Score (1-100)'], 
                                 c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Spending Score (1-100)')
    axes[1, 0].set_title('Clusters: Age vs Spending Score')
    plt.colorbar(scatter3, ax=axes[1, 0])
    
    # 4. Cluster distribution by Gender
    cluster_gender = pd.crosstab(df_viz['Cluster'], df_viz['Gender'])
    cluster_gender.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Cluster Distribution by Gender')
    axes[1, 1].legend(title='Gender')
    axes[1, 1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('clustering_results_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_viz

def analyze_clusters(df_viz):
    """Analyze the characteristics of each cluster"""
    print("\nCluster Analysis:")
    print("=" * 50)
    
    # Group by cluster and calculate statistics
    cluster_stats = df_viz.groupby('Cluster').agg({
        'Age': ['mean', 'std'],
        'Annual Income (k$)': ['mean', 'std'],
        'Spending Score (1-100)': ['mean', 'std'],
        'Gender': lambda x: x.value_counts().to_dict()
    }).round(2)
    
    print("Cluster Statistics:")
    print(cluster_stats)
    
    # More detailed analysis for each cluster
    for cluster in sorted(df_viz['Cluster'].unique()):
        cluster_data = df_viz[df_viz['Cluster'] == cluster]
        print(f"\nCluster {cluster}:")
        print(f"  Size: {len(cluster_data)} customers ({len(cluster_data)/len(df_viz)*100:.1f}%)")
        print(f"  Average Age: {cluster_data['Age'].mean():.1f} ± {cluster_data['Age'].std():.1f}")
        print(f"  Average Annual Income: ${cluster_data['Annual Income (k$)'].mean():.1f}k ± ${cluster_data['Annual Income (k$)'].std():.1f}k")
        print(f"  Average Spending Score: {cluster_data['Spending Score (1-100)'].mean():.1f} ± {cluster_data['Spending Score (1-100)'].std():.1f}")
        print(f"  Gender Distribution: {cluster_data['Gender'].value_counts().to_dict()}")

def create_3d_visualization(df_viz):
    """Create a 3D visualization of the clusters"""
    print("\nCreating 3D visualization...")
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(df_viz['Age'], 
                        df_viz['Annual Income (k$)'], 
                        df_viz['Spending Score (1-100)'],
                        c=df_viz['Cluster'], 
                        cmap='viridis', 
                        alpha=0.7,
                        s=50)
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (k$)')
    ax.set_zlabel('Spending Score (1-100)')
    ax.set_title('3D Visualization of Customer Clusters')
    
    plt.colorbar(scatter)
    plt.savefig('clustering_results_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to execute the clustering pipeline"""
    print("Mall Customer Segmentation using K-Means Clustering")
    print("=" * 55)
    
    # Step 1: Load and explore data
    df = load_and_explore_data()
    
    # Step 2: Preprocess data
    X, X_with_gender, df_processed = preprocess_data(df)
    
    # Step 3: Find optimal number of clusters
    optimal_k, scaler = find_optimal_clusters(X, max_clusters=10)
    
    # Step 4: Perform clustering with optimal k
    cluster_labels, kmeans = perform_clustering(X, optimal_k, scaler)
    
    # Step 5: Visualize results
    df_viz = visualize_clusters(df_processed, X, cluster_labels, optimal_k)
    
    # Step 6: Analyze clusters
    analyze_clusters(df_viz)
    
    # Step 7: Create 3D visualization
    create_3d_visualization(df_viz)
    
    # Additional insights
    print("\nKey Insights:")
    print("=" * 30)
    print("1. The clustering algorithm has segmented customers based on their:")
    print("   - Age")
    print("   - Annual Income")
    print("   - Spending Score")
    print("\n2. This segmentation can help in:")
    print("   - Targeted marketing campaigns")
    print("   - Product recommendations")
    print("   - Customer retention strategies")
    print("   - Pricing strategies")
    
    print(f"\n3. The optimal number of clusters is {optimal_k}, which provides")
    print("   meaningful customer segments for business decisions.")

if __name__ == "__main__":
    main()
