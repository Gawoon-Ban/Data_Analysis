import umap.umap_ as umap
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

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

# Main optimization and clustering function
def main():
    data_path = "survey.csv"  # Dataset path
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

    # Define hyperparameter search space
    param_grid = {
        "umap__n_neighbors": range(5, 700, 10),  # Integers from 5 to 700 with step 10
        "umap__min_dist": np.linspace(0.01, 0.1, 10),  # Floats between 0.01 and 0.1 with 10 steps
        "umap__n_components": range(2, 24, 2),  # Integers from 2 to 24 with step 2
        "clustering__n_clusters": [3, 5, 7, 10],  # Fixed list of cluster counts
        "clustering__gamma": np.linspace(1.0, 2.0, 10),  # Floats between 1.0 and 2.0 with 10 steps
        "clustering__affinity": ["nearest_neighbors", "rbf"],  # Two options
        "clustering__assign_labels": ["kmeans", "discretize", "cluster_qr"]  # Three options
    }

    best_score = -1
    best_params = None
    best_prediction = None

    print("Starting hyperparameter optimization...")
    for params in tqdm(list(ParameterGrid(param_grid))):
        # UMAP transformation
        clusterable_embedding = umap.UMAP(
            n_neighbors=params["umap__n_neighbors"],
            min_dist=params["umap__min_dist"],
            n_components=params["umap__n_components"],
            random_state=42,
            n_jobs=-1
        ).fit_transform(data_transformed)
        
        # Spectral Clustering
        clusterer = SpectralClustering(
            n_clusters=params["clustering__n_clusters"],
            eigen_solver="lobpcg",
            n_init=10,
            gamma=params["clustering__gamma"],
            affinity=params["clustering__affinity"],
            assign_labels=params["clustering__assign_labels"],
            random_state=42
        )

        prediction = clusterer.fit_predict(clusterable_embedding)
        prediction = remap_index(prediction)

        # Evaluate with silhouette score
        if len(np.unique(prediction)) == 1:  # Avoid degenerate cases
            score = -1
        else:
            score = silhouette_score(data_transformed, prediction)

        # Update best parameters
        if score > best_score:
            best_score = score
            best_params = params
            best_prediction = prediction
    
    print(f"Best Silhouette Score: {best_score}")
    print(f"Best Parameters: {best_params}")

    # Save the best prediction
    put_submission_data(best_prediction, "submission_optimized_spectral_clustering.csv")
    print("Optimized submission file 'submission_optimized_spectral_clustering.csv' created.")

if __name__ == '__main__':
    main()
