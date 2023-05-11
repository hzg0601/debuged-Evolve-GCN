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


# class MLP(nn.Module):
#     def __init__(self,
#     out_dim=2,
#     hid_dim=100):
#         super().__init__()

#         self.hid_dim = hid_dim
#         self.activate = torch.relu
#         self.linear2 = nn.Linear(hid_dim,out_dim)

#     def forward(self,inputs):

#         in_dim = inputs.shape[-1]
#         self.linear1 = nn.Linear(in_dim,self.hid_dim)
#         logit = self.linear2(self.activate(self.linear1(inputs)))
#         result = torch.argmax(logit)
#         return result


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
    indexes,labels = dataset.nodes_labels_times[:,0].numpy(), dataset.nodes_labels_times[:,1].numpy()

    nodes_feats = dataset.nodes_feats[indexes].numpy()


    assert args.ml_args["model"] in ["dt","rf","lr","mlp"], "undefined machine learning model"
    assert args.ml_args["feature"] in ["AF","LF","NE","AF+NE","LF+NE"],"unsupported features"
    
    model = eval(args.ml_args["model"])
    if "AF" in args.ml_args["features"]:
        nodes_feats = nodes_feats[:,:94]

    # 如果使用网络嵌入特征
    if "NE" in args.ml_args["feature"]:
        assert args.ml_args['ne'] in ["egcn_h", "egcn_o", "skipfeatsgcn","gcn", "skipgcn","delgcn"], "unsupported nework embedding features"
        file_re = f"^{args.ml_args['ne']}.*{args.data}\.csv\.gz$"
        file_name = log_file + [re.search(file_re,file) for file in os.listdir(log_file)][0]
        ne = pd.read_csv(file_name,header=None, index=None, compression='gzip').to_numpy()[:,1:]
        
        # skipfeatsgcn包含了原始特征
        if  args.ml_args["feature"] == "NE" or args.ml_args['ne'] == "skipfeatsgcn":
            features = ne
        else:
            features = np.concatenate([nodes_feats,ne],axis=1)
    else:
        features = nodes_feats


    train_feats = features[: int((args.train_proportion+args.dev_proportion)*indexes.size(0))]
    train_labels = labels[:int((args.train_proportion+args.dev_proportion)*indexes.size(0))]
    
    test_feats = features[int((args.train_proportion+args.dev_proportion)*indexes.size(0)):]
    test_labels = labels[int((args.train_proportion+args.dev_proportion)*indexes.size(0)):]

    print("loading dataset done, begin to train...")
    model.fit(train_feats,train_labels)
    print("training done,start to eval...")
    result = model.predict(test_feats)
    f1,recall,precision = f1_score(result,test_labels), recall_score(result,test_labels),precision_score(result,test_labels)
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



