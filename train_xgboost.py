import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

# 删掉user_id
sparse_features = ["date", "user_id","product","campaign_id","webpage_id","product_category_id","user_group_id","gender","age_level", "user_depth", "var_1"]
dense_features = ["product_click", "ug_pg_click"]
df = pd.read_csv("./dataset/train_feature_eng.csv")
test_df = pd.read_csv("./dataset/test_feature_eng.csv")

df[sparse_features] = df[sparse_features].fillna(-1, )
df[dense_features] = df[dense_features].fillna(0, )
test_df[sparse_features] = test_df[sparse_features].fillna(-1, )
test_df[dense_features] = test_df[dense_features].fillna(0, )

train_df, valid_df = train_test_split(df, train_size=0.9, random_state=10000)
#X_train = train_df.drop(['isClick','product_click', 'ug_pg_click'], axis=1)
X_train = train_df.drop(['isClick'], axis=1)
Y_train = train_df['isClick']

#X_valid = valid_df.drop(['isClick','product_click', 'ug_pg_click'], axis=1)
X_valid = valid_df.drop(['isClick'], axis=1)
Y_valid = valid_df['isClick']

# model = CatBoostClassifier(iterations=500, 
#                            depth=6,
#                            cat_features=sparse_features,
#                            learning_rate=0.01, 
#                            loss_function='Logloss',
#                            custom_metric="AUC",
#                            use_best_model = True,
#                            task_type="GPU",
#                            logging_level='Verbose')
model = xgb.XGBClassifier(nthread=4, learning_rate=0.1,
                            n_estimators=200, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5)
model.fit(X_train, Y_train)
y_pred = model.predict_proba(X_valid)
print(y_pred[:,1])
y_pred = y_pred[:,1]
val_ruc_auc_score = roc_auc_score(Y_valid, y_pred)
print("ruc_auc_score:{}".format(val_ruc_auc_score))


# X_test = test_df
# pred_ans = model.predict_proba(X_test)[:,1]
# test_df = pd.read_csv("./dataset/test.csv")
# new_df = pd.DataFrame(columns=['id', 'isClick'])
# new_df['id'] = test_df['id']
# new_df['isClick'] = pred_ans
# new_df.to_csv("./processed_data/submission.csv", index=0,encoding='utf-8')

