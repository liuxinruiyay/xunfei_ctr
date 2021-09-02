import pandas as pd
from scipy import sparse
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings('ignore')

def file_preprocess(path):
    df = pd.read_csv(path)
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
    dense_features = ["date_den", "product_category_id_den","user_group_id_den","age_level_den", "user_depth_den", "var_1_den"]
    one_hot_features = ["date", "product","campaign_id","webpage_id", "product_category_id","user_group_id","gender","age_level", "user_depth", "var_1"]
    # 稀疏特征：为类别，表示成id
    # 稠密特征：数值特征进行归一化处理
    df[sparse_features] = df[sparse_features].fillna(-1, )

    #1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    
    print('开始one-hot...')
    for col in one_hot_features:
        onehot_feat = pd.get_dummies(df[col], prefix = col)
        df.drop([col], axis = 1, inplace = True)
        df = pd.concat([df, onehot_feat], axis = 1)
    print('one-hot结束...')

    # mms = MinMaxScaler(feature_range=(0, 1))
    # df[dense_features] = mms.fit_transform(df[dense_features])
    return df, sparse_features, dense_features

label = ["isClick"]
all_data, sparse_features, dense_features = file_preprocess(path="./dataset/all_data.csv")
train_df = all_data[:391825]


test_df = all_data[391825:]
test_df = test_df.drop('isClick', axis=1)


train_df, valid_df = train_test_split(train_df, train_size=0.9, random_state=10000)
X_train = train_df.drop('isClick', axis=1)
Y_train = train_df['isClick']

X_valid = valid_df.drop('isClick', axis=1)
Y_valid = valid_df['isClick']

#model = LogisticRegression(penalty='l2', C=0.01, random_state=40, max_iter=1000, warm_start=True)
model = ExtraTreesClassifier(n_estimators=120, verbose=1)
model.fit(X_train, Y_train)
y_pred = model.predict_proba(X_valid)[:,1]

val_ruc_auc_score = roc_auc_score(Y_valid, y_pred)
print("ruc_auc_score:{}".format(val_ruc_auc_score))