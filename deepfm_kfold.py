import pandas as pd
from scipy import sparse
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold
from deepctr.models import DeepFM, xDeepFM
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

    # 稀疏特征：为类别，表示成id
    # 稠密特征：数值特征进行归一化处理
    df[sparse_features] = df[sparse_features].fillna(-1, )

    #1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    return df, sparse_features

label = ["isClick"]
all_data, sparse_features = file_preprocess(path="./dataset/all_data.csv")
train_df = all_data[:391825]
test_df = all_data[391825:]
test_df = test_df.drop('isClick', axis=1)
# 2.count #unique features for each sparse field,and record dense feature field name
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=all_data[feat].nunique(),embedding_dim=4)
                           for i,feat in enumerate(sparse_features)] 

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
print(feature_names)



test_model_input = {name:test_df[name] for name in feature_names}
total_out = None
kf = KFold(n_splits=20, shuffle=True, random_state=39)
for k, (train_index, valid_index) in enumerate(kf.split(train_df)):
    print("K_Fold_index:"+str(k+1))
    
    # 3.generate input data for model
    train_kf = train_df.iloc[train_index]
    valid_kf = train_df.iloc[valid_index]
    train_model_input = {name:train_kf[name] for name in feature_names}
    valid_model_input = {name:valid_kf[name] for name in feature_names}
    # 4.Define Model,train,predict and evaluate
    model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy", metrics=['accuracy'], )
    history = model.fit(train_model_input, train_kf[label].values,
                            batch_size=256, epochs=1, verbose=2, validation_data=(valid_model_input, valid_kf[label].values), )
    pred_ans = model.predict(valid_model_input, batch_size=256)
    print("test LogLoss:", round(log_loss(valid_kf[label].values, pred_ans), 4))
    print("test AUC:", round(roc_auc_score(valid_kf[label].values, pred_ans), 4))

    test_pred_ans = model.predict(test_model_input, batch_size=256)
    if total_out is None:
        total_out = test_pred_ans
    else:
        total_out = total_out + test_pred_ans


total_out = total_out/20
test_df = pd.read_csv("./dataset/test.csv")
new_df = pd.DataFrame(columns=['id', 'isClick'])
new_df['id'] = test_df['id']
new_df['isClick'] = total_out
new_df.to_csv("./processed_data/submission_xdeepfm_kfold.csv", index=0,encoding='utf-8')
