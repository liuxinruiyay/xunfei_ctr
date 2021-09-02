from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

sparse_features = ["date", "user_id","product","campaign_id","webpage_id","product_category_id","user_group_id","gender","age_level", "user_depth", "var_1"]
dense_features = ["product_click", "ug_pg_click", "camp_click"]
df = pd.read_csv("./dataset/train_feature_eng.csv")
test_df = pd.read_csv("./dataset/test_feature_eng.csv")

df[sparse_features] = df[sparse_features].fillna(-1, )
df[dense_features] = df[dense_features].fillna(0, )
test_df[sparse_features] = test_df[sparse_features].fillna(-1, )
test_df[dense_features] = test_df[dense_features].fillna(0, )

train_df, valid_df = train_test_split(df, train_size=0.9, random_state=10000)
#X_train = train_df.drop(['isClick','ug_pg_click',"gen+age+pro","user+pc"], axis=1)
X_train = train_df.drop(['isClick'], axis=1)
Y_train = train_df['isClick']
#X_valid = valid_df.drop(['isClick','ug_pg_click',"gen+age+pro","user+pc"], axis=1)
X_valid = valid_df.drop(['isClick'], axis=1)
Y_valid = valid_df['isClick']

# print("处理前:",str(X_train.shape))
# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False,max_iter=10000).fit(X_train, Y_train)
# model = SelectFromModel(lsvc, prefit=True)
# X_train_new = model.transform(X_train)
# X_valid_new = model.transform(X_valid)
# print("处理后:",str(X_train_new.shape))

# print("处理前:",str(X_train.shape))
# clf = ExtraTreesClassifier()
# clf = clf.fit(X_train, Y_train)
# print(clf.feature_importances_)  
# select_model = SelectFromModel(clf, prefit=True)
# X_train_new = select_model.transform(X_train)
# X_valid_new = select_model.transform(X_valid)
# print("处理后:",str(X_train_new.shape))

print("处理前:",str(X_train.shape))
X_train_new = SelectKBest(chi2, k=16).fit_transform(X_train, Y_train)
X_valid_new = SelectKBest(chi2, k=16).fit_transform(X_valid, Y_valid)
print("处理后:",str(X_train_new.shape))

model = CatBoostClassifier(iterations=500, 
                           depth=6,
                           
                           learning_rate=0.01, 
                           loss_function='Logloss',
                           custom_metric="AUC",
                           use_best_model = True,
                           task_type="GPU",
                           logging_level='Verbose')
model.fit(X_train_new, Y_train,eval_set=(X_valid_new, Y_valid))
y_pred = model.predict_proba(X_valid_new)
print(y_pred[:,1])
y_pred = y_pred[:,1]
val_ruc_auc_score = roc_auc_score(Y_valid, y_pred)
print("ruc_auc_score:{}".format(val_ruc_auc_score))