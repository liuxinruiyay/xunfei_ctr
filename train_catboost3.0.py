import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
    
    # print('开始one-hot...')
    # for col in one_hot_features:
    #     onehot_feat = pd.get_dummies(df[col], prefix = col)
    #     df.drop([col], axis = 1, inplace = True)
    #     df = pd.concat([df, onehot_feat], axis = 1)
    # print('one-hot结束...')
    #df.to_csv("./processed_data/one-hot.csv")
    return df

# 使用逻辑回归对模型进行训练
sparse_features = ["date", "user_id", "product","campaign_id","webpage_id","product_category_id","user_group_id","gender","age_level", "user_depth", "var_1"]
label = ["isClick"]
all_data = encode(path="./dataset/all_data.csv")
train_df = all_data[:391825]


test_df = all_data[391825:]
test_df = test_df.drop('isClick', axis=1)

train_df, valid_df = train_test_split(train_df, train_size=0.9, random_state=10000)
X_train = train_df.drop('isClick', axis=1)
Y_train = train_df['isClick']

X_valid = valid_df.drop('isClick', axis=1)
Y_valid = valid_df['isClick']

model = CatBoostClassifier(iterations=2000, 
                           depth=6,
                           cat_features=sparse_features,
                           learning_rate=0.04, 
                           l2_leaf_reg=13,
                           one_hot_max_size=80,
                           loss_function='Logloss',
                           custom_metric="AUC",
                           use_best_model = True,
                           task_type="GPU",
                           logging_level='Verbose')
model.fit(X_train, Y_train,eval_set=(X_valid, Y_valid))
print(model.feature_importances_)
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
# new_df.to_csv("./processed_data/submission_catboost.csv", index=0,encoding='utf-8')