import pandas as pd
from scipy import sparse
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

from deepctr.models import DeepFM
from deepctr.models.din import DIN
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("./dataset/train.csv")

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
# df = df.interpolate()

sparse_features = ["date", "user_id", "product","campaign_id","webpage_id","product_category_id","user_group_id","gender","age_level", "user_depth", "var_1"]
# dense_features = []
label = ["isClick"]

# 稀疏特征：为类别，表示成id
# 稠密特征：数值特征进行归一化处理
df[sparse_features] = df[sparse_features].fillna('-1', )
# 进行下采样
print(df[label].value_counts())

under_strat = {0: 50000, 1: 26930}
under = RandomUnderSampler(sampling_strategy=under_strat)
# steps = [('o',over),('u',under)]
steps = [('u', under)]
pipeline = Pipeline(steps=steps)
new_df = pd.DataFrame(columns=sparse_features+label)
new_df[sparse_features], new_df[label] = pipeline.fit_resample(df[sparse_features], df[label])
print(new_df[label].value_counts())
print(new_df[sparse_features].head())

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    new_df[feat] = lbe.fit_transform(new_df[feat])


# 2.count #unique features for each sparse field,and record dense feature field name
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=new_df[feat].nunique(),embedding_dim=128 )
                           for i,feat in enumerate(sparse_features)] 

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model
train, valid = train_test_split(new_df, test_size=0.1, random_state=2018)
train_model_input = {name:train[name] for name in feature_names}
valid_model_input = {name:valid[name] for name in feature_names}

# 4.Define Model,train,predict and evaluate
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
# model = DIN(dnn_feature_columns, history_feature_list, task='binary')
model.compile("adam", "binary_crossentropy",
                  metrics=['accuracy'], )
es = 1
for epoch in range(es):
    print("Epoch:{}".format(epoch+1))
    history = model.fit(train_model_input, train[label].values,
                            batch_size=256, epochs=1, verbose=2, validation_data=(valid_model_input, valid[label].values), )
    pred_ans = model.predict(valid_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(valid[label].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(valid[label].values, pred_ans), 4))

# from tensorflow.python.keras.models import  save_model,load_model
# save_model(model, './deepfm_models/deep_fm_undersample.h5')# save_model, same as before




