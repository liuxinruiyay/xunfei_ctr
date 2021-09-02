import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

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


# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_valid, Y_valid, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 63,
	'num_trees': 100,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'device': 'gpu'
}

# number of leaves,will be used in feature transformation
num_leaf = 63


print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                valid_sets=lgb_train)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict and get data on leaves, training data
y_pred = gbm.predict(X_train,pred_leaf=True)

# feature transformation and write result
print('Writing transformed training data')
transformed_training_matrix = np.zeros([len(y_pred),len(y_pred[0]) * num_leaf],dtype=np.int64)
for i in range(0,len(y_pred)):
	temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
	transformed_training_matrix[i][temp] += 1


# predict and get data on leaves, testing data
y_pred = gbm.predict(X_valid,pred_leaf=True)[:,1]

score = roc_auc_score(Y_valid, y_pred)
print("roc_auc score:{}".format(score))
#feature transformation and write result
# print('Writing transformed testing data')
# transformed_testing_matrix = np.zeros([len(y_pred),len(y_pred[0]) * num_leaf],dtype=np.int64)
# for i in range(0,len(y_pred)):
# 	temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
# 	transformed_testing_matrix[i][temp] += 1

#for i in range(0,len(y_pred)):
#	for j in range(0,len(y_pred[i])):
#		transformed_testing_matrix[i][j * num_leaf + y_pred[i][j]-1] = 1

print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))
print('Feature importances:', list(gbm.feature_importance("gain")))


# # Logestic Regression Start
# print("Logestic Regression Start")

# # load or create your dataset
# print('Load data...')

# # best c = 0.001
# c = np.array([1,0.5,0.1,0.05,0.01,0.005,0.001])
# score = 0
# for t in range(0,len(c)):
#     lm = LogisticRegression(penalty='l2',C=c[t]) # logestic model construction
#     lm.fit(transformed_training_matrix,Y_train)  # fitting the data
#     y_pred_est = lm.predict_proba(transformed_testing_matrix)[:,1]
#     score = roc_auc_score(Y_valid, y_pred_est)
#     print("roc_auc score:{}".format(score))





# X_test = test_df
# pred_ans = model.predict_proba(X_test)[:,1]
# test_df = pd.read_csv("./dataset/test.csv")
# new_df = pd.DataFrame(columns=['id', 'isClick'])
# new_df['id'] = test_df['id']
# new_df['isClick'] = pred_ans
# new_df.to_csv("./processed_data/submission.csv", index=0,encoding='utf-8')

