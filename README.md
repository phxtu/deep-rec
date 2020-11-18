# deep-rec
- 基于tensorflow实现的推荐系统的经典模型
- 依赖
    - python: 3.6.11
    - tensorflow: 1.15.0 gpu
- 数据集：[criteo](http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/)

## 运行
### 准备数据
- 下载数据并解压
- 统计离散特征的出现次数，过滤掉低频特征（次数小于10），保留离散特征列表至`data/criteo/ids.txt`文件下，以便做编码；执行如下命令
    - `cut -f15- /pathOfcriteo/train.txt | sed 's/\t/\n/g' | awk '{if($1!=""){cnt[$1]+=1}}END{for(k in cnt){print k","cnt[k]}}' > data/criteo/ids.cnt`
    - `awk -F, '{if($2>=10){print $1}}' data/criteo/ids.cnt > data/criteo/ids.txt`
- 调用util/split.py将数据文件拆成多个小文件，并将训练文件拆出的小文件按9：1划分为训练集、验证集，将其绝对路径分别写入`data/train.txt`、`data/train.txt`
    - 由于原始训练集的文件很大，有4kw+行，为了加快数据加载数据，这里将文件拆分成多个小文件，以便多进程加载
    - `python util/split.py /pathOfcriteo/train.txt data/criteo/train/part 100`
    - `for f in $(ls data/criteo/train/part*);do echo $(pwd)/$f >> data/files.txt;done`
    - `head -n90 data/files.txt > data/train.txt; tail -n10 data/files > data/val.txt`
### 进行训练
- `python run.py deepfm`
### 执行预测评估
- `python predict.py mdls/deepfm/model.h5`

## 模型
- [DeepFM: Factorization-Machine based Neural Network](https://www.ijcai.org/proceedings/2017/0239.pdf)
    - 在验证集上的结果：`loss:0.4465128124150221, auc:0.8052565`
- [eXtreme Deep Factorization Machine (xDeepFM)](https://arxiv.org/abs/1803.05170)
    - 在验证集上的结果：`loss:0.4437886869439382, auc:0.80759335`
- TODO
    - 其它模型

