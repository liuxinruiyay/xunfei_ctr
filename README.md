# 广告点击率预测  
特征尽量都转换为sparse特征，可以进行one-hot或者embedding，对dense特征进行归一化处理，模型调参使用gridsearchCV进行网格搜索  
1.模型catboost  
2.模型gbdt   
3.Deepctr中deepfm，将one-hot特征转换为embedding  
4.特征工程：用户点击量，商品点击量，不同年龄层级的点击量  
5.模型的K折交叉验证  
6.model stack：使用不同模型的K折输出logits作为特征，输入到第二个模型中（不要与原特征进行组合，有可能会过拟合）  
![image](https://user-images.githubusercontent.com/38974623/131817861-6925f97b-17c7-4bef-a266-87aae91b641e.png)
