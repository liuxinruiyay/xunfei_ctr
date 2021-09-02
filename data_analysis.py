from numpy.core.fromnumeric import product
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

# from deepctr.models import DeepFM
# from deepctr.models.din import DIN
# from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv("./dataset/train.csv")
test_df = pd.read_csv("./dataset/test.csv")
test_df["isClick"] = -1
all_data = train_df.append(test_df)
print(all_data[391824:])
print(all_data.shape) #391825
all_data.to_csv("./dataset/all_data.csv", index=0,encoding='utf-8')
# df = df.drop('id', axis=1)
# sparse_features = ["date", "user_id", "product","campaign_id","webpage_id","product_category_id","user_group_id","gender","age_level", "user_depth", "var_1"]
# label = ["isClick"]


# gender_map = {"Female":0, "Male":1}
# df = df.replace({"gender": gender_map})
# df['date'] = df.date.map(lambda x:int(x.split(' ')[1].split(':')[0]))

# df['gender'] = df.gender.fillna(-1, )
# df['gender'] = df.gender.map(lambda x:int(x))
# df["user_group_id"] = df['user_group_id'].fillna(-1, )
# df["user_group_id"] = df.user_group_id.map(lambda x:int(x))
# df["age_level"] = df['age_level'].fillna(-1, )
# df["age_level"] = df.age_level.map(lambda x:int(x))
# df["user_depth"] = df['user_depth'].fillna(-1, )
# df["user_depth"] = df.user_depth.map(lambda x:int(x))

# # 稀疏特征：为类别，表示成id
# # 稠密特征：数值特征进行归一化处理
# df[sparse_features] = df[sparse_features].fillna(-1, )
# # print(df[df['gender']==1].isClick.value_counts())
# # print(df[df['gender']==0].isClick.value_counts())
# print(df['product'].value_counts())
# print(df['isClick'].value_counts())

# # 新建特征1:product的点击率
# new_df = df.groupby(["product"])["isClick"].value_counts().unstack()
# #new_df['isClick']['product']
# print(new_df)
# click_num_all = df['isClick'].value_counts()[1]
# print(click_num_all)
# # df['product_click_rate'] = df.product_click_rate.map(lambda x: new_df[1][0])
# def cal_click_rate(series):
#     product_id = series['product']
#     click_num = new_df[1][product_id]
#     return click_num/click_num_all
# df['product_click_rate'] = df.apply(cal_click_rate, axis=1)
# print(df[['product','product_click_rate']])
# df.to_csv("./dataset/train_feature_eng.csv")





# for i in range(0,24):
#     print(df[df['date']==i].isClick.value_counts())
# print(df['var_1'].value_counts())
# # 进行下采样
# X = df[sparse_features]
# Y = df[label]
# print(Y.value_counts())

# under_strat = {0: 50000, 1: 26930}
# under = RandomUnderSampler(sampling_strategy=under_strat)
# # steps = [('o',over),('u',under)]
# steps = [('u', under)]
# pipeline = Pipeline(steps=steps)
# # X = np.array(X).reshape(-1,1)
# X_res, Y_res = pipeline.fit_resample(X, Y)
# # X_res = X_res.reshape(-1)
# print(Y_res.value_counts())
# print(X_res.head())


