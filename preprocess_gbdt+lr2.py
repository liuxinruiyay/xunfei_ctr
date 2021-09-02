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
    one_hot_features = ["date", "product","campaign_id","webpage_id", "product_category_id","user_group_id","gender","age_level", "user_depth", "var_1"]
    # 稀疏特征：为类别，表示成id
    # 稠密特征：数值特征进行归一化处理
    df[sparse_features] = df[sparse_features].fillna('-1', )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    print('开始one-hot...')
    for col in one_hot_features:
        onehot_feat = pd.get_dummies(df[col], prefix = col)
        df.drop([col], axis = 1, inplace = True)
        df = pd.concat([df, onehot_feat], axis = 1)
    print('one-hot结束')
    df.to_csv("./processed_data/one-hot.csv")
    return df

# 使用逻辑回归对模型进行训练
df = file_preprocess(path="./dataset/train.csv")
train_df, valid_df = train_test_split(df, train_size=0.9)
X_train = train_df.drop('isClick', axis=1)
Y_train = train_df['isClick']

X_valid = valid_df.drop('isClick', axis=1)
Y_valid = valid_df['isClick']

#X_train, X_train_lr, Y_train, Y_train_lr = train_test_split(X_train, Y_train, test_size=0.9)

# 梯度提升树回归：0.59
print("Start training gbdt.....\n")
grd = GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=200,
    max_depth=4,
    min_samples_split=2,
    verbose=1)
grd_enc = OneHotEncoder()   # onehot
grd_lm = LogisticRegression()   # LR
grd.fit(X_train, Y_train)

grd_enc.fit(grd.apply(X_train)[:, :, 0])
print("Start training lr.....\n")
grd_lm.fit(grd_enc.transform(grd.apply(X_train)[:, :, 0]), Y_train)

valid_pred_ans = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_valid)[:, :, 0]))[:, 1]

score = roc_auc_score(Y_valid, valid_pred_ans)
print("ruc_auc_score:{}".format(score))





#使用训练好的模型生成test数据集结果
test_df = file_preprocess(path="./dataset/test.csv")
test_pred_ans = grd_lm.predict_proba(grd_enc.transform(grd.apply(test_df)[:, :, 0]))[:, 1]

test_df = pd.read_csv("./dataset/test.csv")
new_df = pd.DataFrame(columns=['id', 'isClick'])
new_df['id'] = test_df['id']
new_df['isClick'] = test_pred_ans
new_df.to_csv("./processed_data/submission.csv", index=0,encoding='utf-8')

