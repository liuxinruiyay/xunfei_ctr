import pandas as pd
from scipy import sparse
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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

df = df.drop('id', axis=1)
# df = df.interpolate()

sparse_features = ["date", "user_id", "product","campaign_id","webpage_id","product_category_id","user_group_id","gender"]
dense_features = ["age_level", "user_depth", "var_1"]
label = ["isClick"]

# 稀疏特征：为类别，表示成id
# 稠密特征：数值特征进行归一化处理
df[sparse_features] = df[sparse_features].fillna('-1', )
df[dense_features] = df[dense_features].fillna(0, )

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    df[feat] = lbe.fit_transform(df[feat])
print(df.head())
mms = MinMaxScaler(feature_range=(0, 1))
df[dense_features] = mms.fit_transform(df[dense_features])

# 2.count #unique features for each sparse field,and record dense feature field name

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique(),embedding_dim=4 )
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
# history_feature_list = [SparseFeat(feat, vocabulary_size=df[feat].nunique(),embedding_dim=4 )
#                            for i,feat in enumerate(sparse_features)]
# print(history_feature_list)


feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model
train, valid = train_test_split(df, test_size=0.1, random_state=2018)
train_model_input = {name:train[name] for name in feature_names}
valid_model_input = {name:valid[name] for name in feature_names}

# 4.Define Model,train,predict and evaluate
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
# model = DIN(dnn_feature_columns, history_feature_list, task='binary')
model.compile("adam", "binary_crossentropy",
                  metrics=['accuracy'], )

history = model.fit(train_model_input, train[label].values,
                        batch_size=256, epochs=1, verbose=2, validation_data=(valid_model_input, valid[label].values), )
pred_ans = model.predict(valid_model_input, batch_size=256)
print("test LogLoss", round(log_loss(valid[label].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(valid[label].values, pred_ans), 4))

from tensorflow.python.keras.models import  save_model,load_model
# save_model(model, './deepfm_models/deep_fm.h5')# save_model, same as before




