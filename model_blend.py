import pandas as pd
import os
# deepfm_kfold = pd.read_csv("./processed_data/submission_deepfm_kfold.csv")
# catboost_kfold = pd.read_csv("./processed_data/submission_catboost_kfold.csv")
# gbdt_kfold = pd.read_csv("./processed_data/submission_gbdt_kfold.csv")
# random_forest_kfold = pd.read_csv("./processed_data/submission_random_forest_kfold.csv")
# dcn_kfold = pd.read_csv("./processed_data/submission_dcn_kfold.csv")
# result = deepfm_kfold["isClick"]/3 + catboost_kfold["isClick"]/3 + dcn_kfold["isClick"]/3
result = None
files = os.listdir("./submissions")
size = len(files)
print(size)
for file in files:
    file_path = "./submissions/"+file
    logit = pd.read_csv(file_path)["isClick"]
    if result is None:
        result = logit
    else:
        result = result + logit
result = result/size

print(result)



test_df = pd.read_csv("./dataset/test.csv")
new_df = pd.DataFrame(columns=['id', 'isClick'])
new_df['id'] = test_df['id']
new_df['isClick'] = result
new_df.to_csv("./submissions/catb_gbdt_dfm_lgb.csv", index=0,encoding='utf-8')