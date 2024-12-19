import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
import argparse


def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_data = train_data.dropna(subset=['position'])

    common_features = ["PLAYER_AGE", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", 
                       "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", 
                       "BLK", "TOV", "PF", "PTS"]

    X_train = train_data[common_features]
    y_train = train_data['position']
    X_test = test_data[common_features]
    test_ids = test_data['ID']
    
    return X_train, y_train, X_test, test_ids


def preprocess_data(X_train, X_test):

    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_test.median())
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled


def train_and_evaluate(X_train, y_train, X_test):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  
    }
    f1_scorer = make_scorer(f1_score, average='weighted')
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring=f1_scorer, cv=5)
    grid_search.fit(X_train, y_train)
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)
    return y_pred, best_knn

def save_predictions(ids, predictions, output_file="predictions.csv"):
    result = pd.DataFrame({"ID": ids, "position": predictions})
    result.to_csv(output_file, index=False)

def main(train_path, test_path, output_file):

    X_train, y_train, X_test, test_ids = load_data(train_path, test_path)

    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

    predictions, best_model = train_and_evaluate(X_train_scaled, y_train, X_test_scaled)

    save_predictions(test_ids, predictions, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN Classifier for NBA Player Position Prediction")
    parser.add_argument('--train_path', type=str, default="train.csv", help="Path to the training data CSV file")
    parser.add_argument('--test_path', type=str, default="test.csv", help="Path to the test data CSV file")
    parser.add_argument('--output_file', type=str, default="predictions.csv", help="Path to save the predictions CSV file")
    
    args = parser.parse_args()
    main(args.train_path, args.test_path, args.output_file)