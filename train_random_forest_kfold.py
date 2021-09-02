import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold

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
    #df.to_csv("./processed_data/one-hot.csv")
    return df

# 使用逻辑回归对模型进行训练
label = ["isClick"]
all_data = encode(path="./dataset/all_data.csv")
train_df = all_data[:391825]

test_df = all_data[391825:]
test_df = test_df.drop('isClick', axis=1)


total_out = None
kf = KFold(n_splits=20, shuffle=True, random_state=39)
for k, (train_index, valid_index) in enumerate(kf.split(train_df)):
    print("K_Fold_index:"+str(k+1))
    X_train_kf = train_df.iloc[train_index].drop(['isClick'], axis=1)
    Y_train_kf = train_df['isClick'].iloc[train_index]
    X_valid_kf = train_df.iloc[valid_index].drop(['isClick'], axis=1)
    Y_valid_kf = train_df['isClick'].iloc[valid_index]
    model = RandomForestClassifier(n_estimators=50,
                                    min_samples_split=2,
                                    min_samples_leaf=20,
                                    max_depth=None,
                                    max_features='sqrt',
                                    bootstrap=True,
                                    verbose=1)
    model.fit(X_train_kf, Y_train_kf)
    y_pred = model.predict_proba(X_valid_kf)
    print(y_pred[:,1])
    y_pred = y_pred[:,1]
    val_ruc_auc_score = roc_auc_score(Y_valid_kf, y_pred)
    print("ruc_auc_score:{}".format(val_ruc_auc_score))

    X_test = test_df
    pred_ans = model.predict_proba(X_test)[:,1]
    if total_out is None:
        total_out = pred_ans
    else:
        total_out = total_out + pred_ans

total_out = total_out/20
test_df = pd.read_csv("./dataset/test.csv")
new_df = pd.DataFrame(columns=['id', 'isClick'])
new_df['id'] = test_df['id']
new_df['isClick'] = total_out
new_df.to_csv("./processed_data/submission_random_forest_kfold.csv", index=0,encoding='utf-8')
