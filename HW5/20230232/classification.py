import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def load_data(data_path):
    data = pd.read_csv(data_path)
    data = data.fillna(data.mode().iloc[0])
    return data


def split_feature_label(data):
    feature = data.drop(columns=['position', 'SEASON_ID', 'TEAM_ID', 'GP', 'GS', 'MIN','PTS','PLAYER_AGE','DREB','OREB','FT_PCT','FG3M'])  
    label = data['position']
    return feature, label

def put_submission_data(prediction, label_encoder):
    id_array = np.arange(1, prediction.shape[0] + 1).reshape(-1, 1)  
    decoded_predictions = label_encoder.inverse_transform(prediction)  
    content = np.hstack((id_array, decoded_predictions.reshape(-1, 1)))
    
    df = pd.DataFrame(content, columns=['ID', 'Position'])
    df.to_csv('submission.csv', index=False)

def main():
    global args
    data = load_data(args.train_data_path)
    
    label_encoder = LabelEncoder()
    data['position'] = label_encoder.fit_transform(data['position'])  
    train_feature, train_label = split_feature_label(data)
    
    train_feature, test_feature, train_label, test_label = train_test_split(
        train_feature,
        train_label,
        test_size=args.test_size,
        random_state=20230232,
        stratify=train_label
    )
    
    xgboost_classifier = XGBClassifier(
        booster='gbtree',
        objective='multi:softmax',
        random_state=20230232,
        colsample_bytree=0.9243638650605333,
        colsample_bylevel=0.9,
        gamma=0.0643936232154077,
        max_depth=72,
        min_child_weight=2,
        n_estimators=617,
        learning_rate = 0.09425985908845773,
        subsample = 0.3373069171516473
    )
    
    xgboost_classifier.fit(train_feature, train_label)

    test_prediction = xgboost_classifier.predict(test_feature)
    test_f1_score = f1_score(test_label, test_prediction, average='macro')
    
    submission_data = load_data(args.submission_data_path)
    
    submission_feature = submission_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'ID','PTS','PLAYER_AGE','DREB','OREB','FT_PCT','FG3M'])
    
    submission_prediction = xgboost_classifier.predict(submission_feature)
    
    put_submission_data(submission_prediction, label_encoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default="train.csv")
    parser.add_argument('--submission_data_path', type=str, default="test.csv")
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()

    main()  