"""
预处理eliiptic数据，用于适应elliptic_temporal_dl.py
原数据包含elliptic_txs_classes、elliptic_txs_features、
elliptic_txs_edgelist三个数据，分别表示节点类别、节点特征、边列表
1. classes包含unkown、1、2三种，需要将其转化为-1,1,0；
2. 节点的编号为非连续的长整数，需要重新编码为从0开始的连续整数；
3. features,第一列为节点编号，第二列为时间点，余下列为特征；
4. 
"""
import os 
import pandas as pd
import numpy as np
import regex as re
import tarfile


def preprocess_epllitic(data_dir='./data/elliptic_bitcoin_dataset/',
                        class_file='classes',
                        feature_file='features',
                        edgelist_file='edgelist'):
    print("开始elliptic数据预处理以适应evolvegcn..")
    # 读取数据
    files = os.listdir(data_dir)
    class_df = pd.read_csv([os.path.join(data_dir,file) for file in files if re.search(class_file,file)][0])
    feature_df = pd.read_csv([os.path.join(data_dir,file) for file in files if re.search(feature_file,file)][0],header=None)
    edgelist_df = pd.read_csv([os.path.join(data_dir,file) for file in files if re.search(edgelist_file,file)][0])
    # 1.1 重新编码节点编号
    code_dict ={item: idx for idx, item in enumerate( set(feature_df.iloc[:,0]))}
    orig2contiguos_df = pd.DataFrame({'originalId':code_dict.keys(),
                                      'contigousId':code_dict.values()})
    # 将第一、第二列变为float类型
    feature_df.iloc[:,0] = feature_df.iloc[:,0].apply(
        lambda x: code_dict[x]).astype('float')
    feature_df.iloc[:,1] = feature_df.iloc[:,1].astype('float')
    # 2. 重新编码class的节点ID，和类别
    
    class_df.iloc[:,0] = class_df.iloc[:,0].apply(lambda x: code_dict[x])
    
    label_dict = {"unknown":-1.0,'1':1.0,'2':0.0}
    
    class_df.iloc[:,1] = class_df.iloc[:,1].apply(lambda x: label_dict[x])
    # 3 取出新的节点编号和时间点
    nodetime_df = feature_df.iloc[:,[0,1]]
    # 修改nodetime_df的列名
    nodetime_df.columns = ["txId",'timestamp']
    
    # 4. 节点特征中的时间点是发生交易的时间点，因此发生交易的两个节点的时间点一致
    # 只需将nodetime_df左连接到edge_list上就可以得到带时间步的edge_list
    edgelist_df_timed = pd.merge(edgelist_df,nodetime_df,
                                 how='left',left_on='txId1',right_on='txId')
    # 5. 定义各文件的文件名
    file_names = ["./data/elliptic_temporal/elliptic_txs_features.csv",
                  "./data/elliptic_temporal/elliptic_txs_classes.csv",
                  "./data/elliptic_temporal/elliptic_txs_edgelist.csv"]
    # 写入本地文件
    if not os.path.exists('./data/elliptic_temporal'):
        os.mkdir('./data/elliptic_temporal/')
    feature_df.to_csv(file_names[0],index=False)
    class_df.to_csv(file_names[1],index=False)
    edgelist_df_timed.to_csv(file_names[2],index=False)
    # 6. 打包维elliptic_bitcoin_datset.tar.gz文件
    # 打包完就删除
    tar = tarfile.open("./data/elliptic_temporal/elliptic_bitcoin_dataset_cont.tar.gz",'w:gz')

    for file in file_names:
        tar.add(file)
        os.remove(file)
    tar.close()
    print("done!")
    # 7. 删除文件
    
    
if __name__ == "__main__":
    preprocess_epllitic()