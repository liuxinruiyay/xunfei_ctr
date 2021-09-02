import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import numpy as np

def file_preprocess(path):
    df = pd.read_csv(path)
    gender_map = {"Female":0, "Male":1}
    df = df.replace({"gender": gender_map})
    df['date'] = df.date.map(lambda x:int(x.split(' ')[1].split(':')[0]))
    df['gender'] = df.gender.fillna('-1', )
    df['gender'] = df.gender.map(lambda x:int(x))
    df["user_group_id"] = df['user_group_id'].fillna('-1', )
    df["user_group_id"] = df.user_group_id.map(lambda x:int(x))
    df["age_level"] = df['age_level'].fillna('-1', )
    df["age_level"] = df.age_level.map(lambda x:int(x))
    df["user_depth"] = df['user_depth'].fillna('-1', )
    df["user_depth"] = df.user_depth.map(lambda x:int(x))
    df = df.drop('id', axis=1)

    sparse_features = ["date", "user_id", "product","campaign_id","webpage_id","product_category_id","user_group_id","gender","age_level", "user_depth", "var_1"]
    # 稀疏特征：为类别，表示成id
    # 稠密特征：数值特征进行归一化处理
    df[sparse_features] = df[sparse_features].fillna('-1', )
    # df[dense_features] = df[dense_features].fillna(0, )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    return df

# 使用逻辑回归对模型进行训练
df = file_preprocess(path="./dataset/train.csv")
train_df, valid_df = train_test_split(df, train_size=0.9)
X_train = train_df.drop('isClick', axis=1)
Y_train = train_df['isClick']

X_valid = valid_df.drop('isClick', axis=1)
Y_valid = valid_df['isClick']

# 梯度提升树回归：0.59
print("Start training gbdt.....\n")
gbdt = GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=1,
    max_depth=4,
    min_samples_split=2,
    verbose=1)
gbdt.fit(X_train, Y_train)

X_train_gbdt = gbdt.apply(X_train)[:, :, 0]
X_valid_gbdt = gbdt.apply(X_valid)[:, :, 0]
gb_onehot = OneHotEncoder()
gb_onehot.fit(gbdt.apply(X_train)[:, :, 0])
X_train_gbdt_onehot = gb_onehot.fit_transform(X_train_gbdt)
X_valid_gbdt_onehot = gb_onehot.fit_transform(X_valid_gbdt)

print("Start training lr.....\n")
lr = LogisticRegression()
lr.fit(X_train_gbdt_onehot, Y_train)
pred_ans = lr.predict_proba(X_valid_gbdt_onehot)[:, 1]
print(pred_ans.shape)
score = roc_auc_score(Y_valid, lr.predict_proba(X_valid_gbdt_onehot)[:, 1])
print("ruc_auc_score:{}".format(score))




#使用训练好的模型生成test数据集结果
test_df = file_preprocess(path="./dataset/test.csv")
X_test = test_df
X_test_gbdt = gbdt.apply(X_test)[:, :, 0]
X_test_gbdt_onehot = gb_onehot.fit_transform(X_test_gbdt)
pred_ans = lr.predict_proba(X_test_gbdt_onehot)[:, 1]


test_df = pd.read_csv("./dataset/test.csv")
new_df = pd.DataFrame(columns=['id', 'isClick'])
new_df['id'] = test_df['id']
new_df['isClick'] = pred_ans
new_df.to_csv("./processed_data/submission.csv", index=0,encoding='utf-8')
# df.to_csv("./processed_data/processed_train.csv")
# print(df.isna().iloc[8])
