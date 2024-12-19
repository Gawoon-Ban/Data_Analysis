import umap.umap_ as umap
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load dataset
def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

# Save submission data
def put_submission_data(prediction, file_name):
    id_array = np.arange(0, len(prediction)).reshape(-1, 1)  # Adjust ID range based on dataset size
    content = np.hstack((id_array, prediction.reshape(-1, 1)))
    df = pd.DataFrame(content, columns=['ID', 'Label'])
    df.to_csv(file_name, index=False)

# Remap index to sequential labels
def remap_index(prediction):
    _, idx = np.unique(prediction, return_index=True)
    list_of_index = prediction[np.sort(idx)]
    mapping = {list_of_index[i]: i for i in range(len(list_of_index))}
    return np.vectorize(mapping.__getitem__)(prediction)

# Main clustering function
def main():
    global args
    
    data_path = args.data_path
    data = load_data(data_path)

    # Separate features and labels (if available) or columns with strings
    categorical_features = ['education', 'education level', 'marital status', 'Occupation', 'relationship', 'race', 'sex', 'native country', 'income']
    numerical_features = [col for col in data.columns if col not in categorical_features + ['ID']]  # Assuming 'ID' is the identifier column
    
    # Preprocessing pipeline
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Use ColumnTransformer to apply transformations to the columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Apply transformations to the data
    data_transformed = preprocessor.fit_transform(data)
    
    print("UMAP running...")
    clusterable_embedding = umap.UMAP(
        n_neighbors=350,  # Default neighborhood size
        min_dist=0.06072236268436812,    # Default minimum distance
        n_components=14,  # Reduce to 2D for visualization
        n_jobs=-1
    ).fit_transform(data_transformed)
    
    # Spectral Clustering
    print("Clustering running...")
    clusterer = SpectralClustering(
        n_clusters=3,  # Adjust this based on dataset
        eigen_solver="lobpcg",
        n_init=18,
        gamma=1.7208661160648056,
        affinity="rbf",
        assign_labels="cluster_qr",
        degree=4,
        coef0=0.29616096740848313,
        n_neighbors=550,
        random_state=42
    )
    prediction = clusterer.fit_predict(clusterable_embedding)
    prediction = remap_index(prediction)

    # Save the submission
    put_submission_data(prediction, "submission_spectral_clustering.csv")
    print("Submission file 'submission_spectral_clustering.csv' created.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', type=str, default="survey.csv")
    global args
    args = parser.parse_args()
    main()
