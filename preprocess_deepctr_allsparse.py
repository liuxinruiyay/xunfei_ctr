import pandas as pd
from scipy import sparse
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from deepctr.models import DeepFM, xDeepFM, DCN
from deepctr.models.din import DIN
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
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
    # 稀疏特征：为类别，表示成id
    # 稠密特征：数值特征进行归一化处理
    df[sparse_features] = df[sparse_features].fillna(-1, )

    #1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    
    # mms = MinMaxScaler(feature_range=(0, 1))
    # df[dense_features] = mms.fit_transform(df[dense_features])
    return df, sparse_features, dense_features

label = ["isClick"]
all_data, sparse_features, dense_features = file_preprocess(path="./dataset/all_data.csv")
train_df = all_data[:391825]

# #-------------处理数据不均衡------------------
# under_strat = {0: 50000, 1: 26930}
# under = RandomUnderSampler(sampling_strategy=under_strat)
# # steps = [('o',over),('u',under)]
# steps = [('u', under)]
# pipeline = Pipeline(steps=steps)
# new_train_df = pd.DataFrame(columns=sparse_features+label)
# new_train_df[sparse_features], new_train_df[label] = pipeline.fit_resample(train_df[sparse_features], train_df[label])
# print(new_train_df[label].value_counts())
# print(new_train_df[sparse_features])
# #-------------处理数据不均衡------------------
# train_df["date_den"] = train_df["date"]
# train_df["product_category_id_den"] = train_df["product_category_id"]
# train_df["user_group_id_den"] = train_df["user_group_id"]
# train_df["age_level_den"] = train_df["age_level"]
# train_df["user_depth_den"] = train_df["user_depth"]
# train_df["var_1_den"] = train_df["var_1"]
# mms = MinMaxScaler(feature_range=(0, 1))
# train_df[dense_features] = mms.fit_transform(train_df[dense_features])


test_df = all_data[391825:]
test_df = test_df.drop('isClick', axis=1)
# 2.count #unique features for each sparse field,and record dense feature field name
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=all_data[feat].nunique(),embedding_dim=4)
                           for i,feat in enumerate(sparse_features)]
# fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=all_data[feat].nunique(),embedding_dim=4)
#                            for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
#                           for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
print(feature_names)
# 3.generate input data for model
train, valid = train_test_split(train_df, test_size=0.1, random_state=2018)
train_model_input = {name:train[name] for name in feature_names}
valid_model_input = {name:valid[name] for name in feature_names}
test_model_input = {name:test_df[name] for name in feature_names}

# 4.Define Model,train,predict and evaluate
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')

model.compile("adam", "binary_crossentropy", metrics=['accuracy'], )
es = 1
for epoch in range(es):
    print("Epoch:{}".format(epoch))
    history = model.fit(train_model_input, train[label].values,
                            batch_size=256, epochs=1, verbose=2, validation_data=(valid_model_input, valid[label].values), )
    pred_ans = model.predict(valid_model_input, batch_size=256)
    test_pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss:", round(log_loss(valid[label].values, pred_ans), 4))
    print("test AUC:", round(roc_auc_score(valid[label].values, pred_ans), 4))

# test_df = pd.read_csv("./dataset/test.csv")
# print(len(test_df))
# print(test_pred_ans)
# new_df = pd.DataFrame(columns=['id', 'isClick'])
# new_df['id'] = test_df['id']
# new_df['isClick'] = test_pred_ans
# new_df.to_csv("./processed_data/submission_deepfm_under_sample.csv", index=0,encoding='utf-8')





