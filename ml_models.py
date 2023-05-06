from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score,recall_score, precision_score
import torch.nn as nn
import torch.functional as F 
import torch
import os 
import regex as re 
import pandas as pd 
import numpy as np 


import utils as u
from run_exp import build_dataset,build_random_hyper_params,build_tasker
import splitter as sp

log_file = './embs/'

# import argparse
# parser = argparse.ArgumentParser(description="Machine Learning Methods for Illicit Node Classification")

# parser.add_argument("--model",type=str,default="mlp",choices=["dt","rf","lr","mlp"])
# parser.add_argument("--feature",type=str,default="AF+NE",choices=["AF","LF","NE","AF+NE","LF+NE"])


class MLP(nn.Module):
    def __init__(self,
    out_dim=2,
    hid_dim=100):
        super().__init__()

        self.hid_dim = hid_dim
        self.activate = torch.relu
        self.linear2 = nn.Linear(hid_dim,out_dim)

    def forward(self,inputs):

        in_dim = inputs.shape[-1]
        self.linear1 = nn.Linear(in_dim,self.hid_dim)
        logit = self.linear2(self.activate(self.linear1(inputs)))
        result = torch.argmax(logit)
        return result


dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)
lr = LogisticRegression(penalty="l1")
# mlp = MLP(hid_dim=100,out_dim=2)
mlp = MLPClassifier(hidden_layer_sizes=100,learning_rate="adaptive")


def main():

    print("start to eval model with machine learning method...")

    parser = u.create_parser()
    args = u.parse_args(parser)
    args = build_random_hyper_params(args)
    if args.task != "static_node_cls":
        args.task = 'static_node_cls'
	#build the dataset
    dataset = build_dataset(args)
	#build the tasker
    tasker = build_tasker(args,dataset)
    tasker.is_static = True
	#build the splitter
    splitter = sp.splitter(args,tasker)

    assert args.ml_args.model in ["dt","rf","lr","mlp"], "undefined machine learning model"
    assert args.ml_args.feature in ["AF","LF","NE","AF+NE","LF+NE"],"unsupported features"
    
    model = eval(args.ml_args.model)

    train = list(splitter.train)[0].numpy() # train,dev,test are DataLoaders with only 1 batch
    dev  = list(splitter.dev)[0].numpy()
    test  = list(splitter.test)[0].numpy()
    train_idx = splitter.train_idx.numpy()
    dev_idx = splitter.dev_idx.numpy()
    test_idx = splitter.test_idx.numpy()

    # 如果使用LF即local feature则取前94个
    if "LF" in args.ml_args.feature:
        train,dev,test = train[:,:94],dev[:,:94], test[:,:94]
    # 如果使用网络嵌入特征
    if "NE" in args.ml_args.feature:
        assert args.ml_args.ne in ["egcn_h", "egcn_o", "skipfeatsgcn","gcn", "skipgcn"], "unsupported nework embedding features"
        file_name = log_file + [re.search(f"^{args.ml_args.ne}",file) for file in os.listdir(log_file)][0]
        ne = pd.read_csv(file_name,compression="gzip").to_numpy()
        train_ne = ne[train_idx]
        dev_ne = ne[dev_idx]
        test_ne = ne[test_idx]
        # skipfeatsgcn包含了原始特征
        if  args.ml_args.feature == "NE" or args.ml_args.ne == "skipfeatsgcn":
            train,dev,test = train_ne, dev_ne, test_ne
        # 如果只使用嵌入或ne为skipfeatsgcn的嵌入向量，则将ne的值作为特征，否则将ne的结果与原始特征合并
        else:
            train = np.concatenate([train,train_ne],axis=1)
            dev = np.concatenate([dev,dev_ne],axis=1)
            test = np.concatenate([test,test_ne],axis=1)
    
    train_feat = np.concatenate([train,dev],axis=0)

    train_label,dev_label,test_label = dataset.label[train_idx],dataset.label[dev_idx],dataset.label[test_idx]

    train_label = np.concatenate([train_label,dev_label],axis=-1)
    print("loading dataset done, begin to train...")
    model.fit(train_feat,train_label)
    print("training done,start to eval...")
    result = model.predict(test)
    f1,recall,precision = f1_score(result,test_label), recall_score(result,test_label),precision(result,test_label)
    print("eval done.")

    print(f"""
    the f1,recall, and precision of model:{args.ml_args.model} 
    under feature: {args.ml_args.feature} and
     ne: {args.ml_args.ne} 
    is {f1},{recall},{precision}
    """)

    print("done!")


if __name__ == "__main__":
    main()



