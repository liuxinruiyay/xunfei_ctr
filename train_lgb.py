import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np


def encode(path):
    df = pd.read_csv(path)
    print(df.info())
    gender_map = {"Female":0, "Male":1}
    df = df.replace({"gender": gender_map})
    df['date'] = df.date.map(lambda x:int(x.split(' ')[1].split(':')[0]))
    df['gender'] = df.gender.fillna(-1, )
    df['gender'] = df.gender.map(lambda x:int(x))
    df["user_group_id"] = df['user_group_id'].fillna(-1, )
    df["user_group_id"] = df.user_group_id.map(lambda x:int(x))
    df["age_level"] = df['age_level'].fillna(-1, )
    df["age_level"] = df.age_level.map(lambda x:int(x))
    df["user_depth"] = df['user_depth'].fillna(-1, )
    df["user_depth"] = df.user_depth.map(lambda x:int(x))
    df = df.drop('id', axis=1)

    sparse_features = ["date", "user_id", "product","campaign_id","webpage_id","product_category_id","user_group_id","gender","age_level", "user_depth", "var_1"]
    one_hot_features = ["date", "product","campaign_id","webpage_id", "product_category_id","user_group_id","gender","age_level", "user_depth", "var_1"]
    #dense_features = ["product_click", "ug_pg_click"]
    # 稀疏特征：为类别，表示成id
    # 稠密特征：数值特征进行归一化处理
    df[sparse_features] = df[sparse_features].fillna(-1, )
    #df[dense_features] = df[dense_features].fillna(0, )
    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    # mms = MinMaxScaler(feature_range=(0, 1))
    # df[dense_features] = mms.fit_transform(df[dense_features])
    
    print('开始one-hot...')
    for col in one_hot_features:
        onehot_feat = pd.get_dummies(df[col], prefix = col)
        df.drop([col], axis = 1, inplace = True)
        df = pd.concat([df, onehot_feat], axis = 1)
    print('one-hot结束...')
    # #df.to_csv("./processed_data/one-hot.csv")
    return df

label = ["isClick"]
all_data = encode(path="./dataset/all_data.csv")
train_df = all_data[:391825]


test_df = all_data[391825:]
test_df = test_df.drop('isClick', axis=1)

train_df, valid_df = train_test_split(train_df, train_size=0.9, random_state=12000)
X_train = train_df.drop('isClick', axis=1)
Y_train = train_df['isClick']

X_valid = valid_df.drop('isClick', axis=1)
Y_valid = valid_df['isClick']

# 1.learning_rate && n_estimators:98
# params = {
#     'boosting_type': 'gbdt', 
#     'objective': 'regression', 
#     'learning_rate': 0.1, 
#     'num_leaves': 50, 
#     'max_depth': 6,
#     'subsample': 0.8, 
#     'colsample_bytree': 0.8, 
#     }
# data_train = lgb.Dataset(X_train, Y_train, silent=True)
# cv_results = lgb.cv(
#     params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
#     early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

# print('best n_estimators:', len(cv_results['rmse-mean']))
# print('best cv score:', cv_results['rmse-mean'][-1])

# 2.max_depth:5 && num_leaves:50
# model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
#                               learning_rate=0.1, n_estimators=98, max_depth=6,
#                               metric='rmse', bagging_fraction = 0.8,feature_fraction = 0.8)

# params_test1={
#     'max_depth': range(3, 8, 2),
#     'num_leaves':range(50, 170, 30)
# }
# gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
# gsearch1.fit(X_train, Y_train)
# print(gsearch1.best_score_)
# print(gsearch1.best_params_)

# 3.min_child_samples:22 && min_child_weights:0.001
# params_test3={
#     'min_child_samples': [18, 19, 20, 21, 22],
#     'min_child_weight':[0.001, 0.002]
# }
# model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
#                               learning_rate=0.1, n_estimators=98, max_depth=5, 
#                               metric='rmse', bagging_fraction = 0.8, feature_fraction = 0.8)
# gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
# gsearch3.fit(X_train, Y_train)
# print(gsearch3.best_score_)
# print(gsearch3.best_params_)

# # 4.feature_fraction:0.8 && bagging_fraction:0.9
# params_test4={
#     'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
#     'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
# }
# model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
#                               learning_rate=0.1, n_estimators=98, max_depth=5, 
#                               metric='rmse', bagging_freq = 5,  min_child_samples=22)
# gsearch4 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
# gsearch4.fit(X_train, Y_train)
# print(gsearch4.best_score_)
# print(gsearch4.best_params_)


# 5.reg_alpha:0 && reg_lamda:0.5
# params_test6={
#     'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
#     'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]
# }
# model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
#                               learning_rate=0.1, n_estimators=98, max_depth=5, 
#                               metric='rmse',  min_child_samples=22, feature_fraction=0.8,
#                               bagging_fraction=0.9)
# gsearch6 = GridSearchCV(estimator=model_lgb, param_grid=params_test6, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
# gsearch6.fit(X_train, Y_train)
# print(gsearch6.best_score_)
# print(gsearch6.best_params_)

# model_lgb = lgb.LGBMClassifier(num_leaves=50,
#                               learning_rate=0.01, n_estimators=150, max_depth=5, 
#                               metric='binary_logloss',  min_child_samples=22, feature_fraction=0.8,
#                               bagging_fraction=0.9, reg_lambda=0.5,verbose=1, device='gpu')
model_lgb = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, 
                            learning_rate=0.07, n_estimators=300, subsample_for_bin=200000, 
                            objective='binary', class_weight={0:0.5, 1:0.5}, min_split_gain=0.0, 
                            min_child_weight=0.001, min_child_samples=20, subsample=1.0,
                            subsample_freq=0, colsample_bytree=1.0, reg_alpha=17, 
                            reg_lambda=130, random_state=None, n_jobs=8, 
                            silent=True, importance_type='split')
model_lgb.fit(X_train, Y_train)
y_pred = model_lgb.predict_proba(X_valid)[:,1]
print(y_pred)
val_ruc_auc_score = roc_auc_score(Y_valid, y_pred)
print("ruc_auc_score:{}".format(val_ruc_auc_score))


X_test = test_df
pred_ans = model_lgb.predict_proba(X_test)[:,1]
test_df = pd.read_csv("./dataset/test.csv")
new_df = pd.DataFrame(columns=['id', 'isClick'])
new_df['id'] = test_df['id']
new_df['isClick'] = pred_ans
new_df.to_csv("./processed_data/submission_lgb.csv", index=0,encoding='utf-8')




