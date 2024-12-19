import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import argparse

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    if 'MIN' in train_data.columns:
        train_data['MIN'] = train_data['MIN'].fillna(train_data['MIN'].median())
    
    common_features = ["PLAYER_AGE", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", 
                       "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", 
                       "BLK", "TOV", "PF", "PTS"]
    
    X_train = train_data[common_features]
    y_train = train_data['MIN']
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
        'max_depth': [3, 5, 10, None],              
        'min_samples_split': [2, 5, 10],              
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, scoring='neg_mean_absolute_error', cv=5)
    grid_search.fit(X_train, y_train)
    best_tree_model = grid_search.best_estimator_
    y_pred = best_tree_model.predict(X_test)
    
    return y_pred

def save_predictions(ids, predictions, output_file="predictions.csv"):
    result = pd.DataFrame({"ID": ids, "MIN": predictions})
    result.to_csv(output_file, index=False)

def main(train_path, test_path, output_file):
    X_train, y_train, X_test, test_ids = load_data(train_path, test_path)

    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
    
    predictions = train_and_evaluate(X_train_scaled, y_train, X_test_scaled)
    
    save_predictions(test_ids, predictions, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DT for regre for MIN")
    parser.add_argument('--train_path', type=str, default="train.csv")
    parser.add_argument('--test_path', type=str, default="test.csv")
    parser.add_argument('--output_file', type=str, default="predictions.csv")
    
    args = parser.parse_args()
    main(args.train_path, args.test_path, args.output_file) 