import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from scipy.stats import uniform, randint

# 데이터 불러오기
train = pd.read_csv('train.csv')
train = train.fillna(train.mode().iloc[0])
test = pd.read_csv('test.csv')
test = test.fillna(test.mode().iloc[0])

# feature와 target 분리
X = train[['FGM', 'FGA', 'FG_PCT', 
           'FG3M', 'FG3_PCT', 'FTA',  'OREB', 
            'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']].copy()
y = train['MIN']

# Test 데이터의 feature 가져오기
X_test = test[['FGM', 'FGA', 'FG_PCT', 
               'FG3M', 'FG3_PCT', 'FTA',  'OREB', 
                'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']].copy()

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 하이퍼파라미터 탐색 범위 정의
param_distributions = {
    'C': randint(1,1000),  # C는 1부터 1000까지 100개의 실수
    'gamma': uniform(0.001, 0.2),  # gamma는 0.001부터 0.1까지 50개의 실수
    'kernel': ['rbf']  # 고정값
}

# RandomizedSearchCV 실행
random_search = RandomizedSearchCV(
    SVR(),
    param_distributions=param_distributions,
    n_iter=50,  # 탐색할 랜덤 조합의 수
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1,
    random_state=42  # 결과 재현성을 위해 설정
)

random_search.fit(X_scaled, y)

# 최적의 파라미터 출력
print("Best parameters found: ", random_search.best_params_)

# 최적 모델로 예측
best_model = random_search.best_estimator_
predictions = best_model.predict(X_test_scaled)

# 결과 저장
results = pd.DataFrame({
    'ID': test['ID'],
    'MIN': predictions
})

results.to_csv('submission.csv', index=False)
print("Results saved to submission.csv")
