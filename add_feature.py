from numpy.core.fromnumeric import prod, product
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
    df = df.fillna(-1, )
    sparse_features = ["date", "user_id", "product","campaign_id","webpage_id","product_category_id","user_group_id","gender","age_level", "user_depth", "var_1"]
    df[sparse_features] = df[sparse_features].fillna(-1, )
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     df[feat] = lbe.fit_transform(df[feat])
    return df

train_df = file_preprocess("./dataset/train.csv")
test_df = file_preprocess("./dataset/test.csv")
click_num_all = train_df['isClick'].value_counts()[1]
print(train_df.info())


# 新建特征1:product, isClick
new_df = train_df.groupby(["product"])["isClick"].value_counts().unstack()
#new_df['isClick']['product']
print(new_df)
def cal_pro_click(series):
    product_id = series['product']
    click_num = new_df[1][product_id]
    pro_click_all  = new_df[1][product_id] + new_df[0][product_id]
    return pro_click_all/click_num
train_df['product_click'] = train_df.apply(cal_pro_click, axis=1)
test_df['product_click'] = test_df.apply(cal_pro_click, axis=1)
print(train_df[['product','product_click']])


# 新建特征2:user_group_id, product_category_id, isClick
new_df = train_df.groupby(["user_group_id", "product_category_id"])["isClick"].value_counts().unstack()
#new_df['isClick']['user_group_id']['product_category_id']
print(new_df)
#print(new_df[1][-1][0])
def cal_ug_pg_click(series):
    product_category_id = series['product_category_id']
    user_group_id = series['user_group_id']
    click_num = new_df[1][user_group_id][product_category_id]
    #print(click_num)
    return click_num
train_df['ug_pg_click'] = train_df.apply(cal_ug_pg_click, axis=1)
test_df['ug_pg_click'] = test_df.apply(cal_ug_pg_click, axis=1)
print(train_df[['product_category_id','user_group_id','ug_pg_click']])


# 新建特征3:gender, age_level, product
train_df['gen+age+pro'] = train_df['gender'].map(str)+"_"+train_df['age_level'].map(str)+"_"+train_df['product'].map(str)
test_df['gen+age+pro'] = test_df['gender'].map(str)+"_"+test_df['age_level'].map(str)+"_"+test_df['product'].map(str)
lbe = LabelEncoder()
train_df['gen+age+pro'] = lbe.fit_transform(train_df['gen+age+pro'])
test_df['gen+age+pro'] = lbe.fit_transform(test_df['gen+age+pro'])
print(train_df['gen+age+pro'].value_counts())


# 新建特征4:user_id, product_category_id
train_df['user+pc'] = train_df['user_id'].map(str)+"_"+train_df['product_category_id'].map(str)
test_df['user+pc'] = test_df['user_id'].map(str)+"_"+test_df['product_category_id'].map(str)
lbe = LabelEncoder()
train_df['user+pc'] = lbe.fit_transform(train_df['user+pc'])
test_df['user+pc'] = lbe.fit_transform(test_df['user+pc'])
print(train_df['user+pc'].value_counts())

# 新建特征5:campgain_id, isClick
new_df = train_df.groupby(["campaign_id"])["isClick"].value_counts().unstack()
#new_df['isClick']['campaign_id']
print(new_df)
def cal_camp_click(series):
    campaign_id = series['campaign_id']
    click_num = new_df[1][campaign_id]
    camp_click_all  = new_df[1][campaign_id] + new_df[0][campaign_id]
    return click_num/camp_click_all
train_df['camp_click'] = train_df.apply(cal_camp_click, axis=1)
test_df['camp_click'] = test_df.apply(cal_camp_click, axis=1)
print(train_df[['campaign_id','camp_click']])

print(train_df.info())
train_df.to_csv("./dataset/train_feature_eng.csv", index=0)
test_df.to_csv("./dataset/test_feature_eng.csv", index=0)