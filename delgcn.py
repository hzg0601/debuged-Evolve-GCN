import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import torch.functional as F 
from typing import Union

"""
脚本实际上执行的H版本的第二种实现
给定邻接矩阵列表A_list，原始节点特征矩阵X_0：
1 根据预定的最大样本量参数num_nodes，构造每个邻接矩阵的掩码向量序列node_mask_list，
以X_0为Node_list的初始元素，以A_list, Node_list, node_mask_list为输入，预定义每一层的
嵌入向量维度，与初始节点特征矩阵X_0,作为特征维度列表参数feats,本脚本的GRCU层数是
根据feats的长度定义好的；
2. 在第一层GRCU中，以节点的原始嵌入向量、随机初始化的GCN参数向量、节点掩码向量node_mask_list为输入：
    2.1 首先根据节点的原始嵌入向量、节点掩码向量选出前topk个节点的特征向量，组成一个feats[0]\times feats[1]的矩阵
    然后执行GRU，得到一个feats[0]\times feats[1]的GCN参数矩阵，执行GCN卷积，得到num\times feats[1]的嵌入向量，
    2.2 遍历A_list、node_mask_list，得到t个GCN参数矩阵列表和t个嵌入矩阵列表；
3. 在余下层，保留学习的嵌入矩阵列表，但重新初始化GCN的参数层。
4. 最终得到最后一层GRCU的嵌入向量列表，以最后一个时间点的嵌入为节点的嵌入向量。

存在如下缺陷：
1. 利用每个时间点逐步回归，计算效率低；
2. !~每层的GRCU训练中，都重新对GCN进行初始化，没有充分利用学习的结果；
3. 学者业已证明，gcn中激活函数和权重矩阵对模型表现的贡献较低，但却大大增加了计算量；
4. 每次更新中仅使用一层gcn来生成嵌入表示，学习到的hop较少

解决方案：
1. 使用informer作为时间序列学习组件；
2. 引入权重共享机制；
3. 使用lightgcn、gat等作为gcn模型

动态图神经网络的出发点是根据历史邻接矩阵和历史节点特征预测下一个（或下几个）时间点的节点嵌入矩阵
节点嵌入矩阵是可观测的，学习的嵌入矩阵作为隐状态而存在，
但将节点的嵌入矩阵作为隐状态对于较大的网络而言是不可实现的，
因此折中的方案是将图神经网络的参数作为隐状态

对于每一个时点，利用历史的邻接矩阵和节点特征向量，基于LightGCN进行嵌入向量计算，
vae\gan\diffusion\transformer来更新LightGCN的权重，然后利用更新的权重计算嵌入向量，
利用嵌入向量进行下游任务计算。
两种方式：
1. 逐步回归，每次更新一个时间步的权重和嵌入向量，使用最后一个时间步的嵌入向量进行下游任务；
2. 一次更新所有权重，使用全部历史嵌入向量进行下游任务
"""

class DELGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False):
        super().__init__()
        delgcn_args = u.Namespace({})
        # 规定每一层的特征的维度
        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.delgcn_layers = []
        self._parameters = nn.ParameterList()
        # 似乎只有3层
        for i in range(1,len(feats)):

            delgcn_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation,
                                     "n_heads":args.n_heads,
                                     "de_layers":args.de_layers,
                                     "lgcn_layers":args.lgcn_layers,
                                     "keep_prob":args.keep_prob,
                                     "device": device
                                     })

            delgcn = DELGCNLayer(delgcn_args)
            #print (i,'delgcn', delgcn)
            self.delgcn_layers.append(delgcn.to(self.device))
            self._parameters.extend(list(self.delgcn_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list):
        node_feats= Nodes_list[-1]
        # # only one-layer
        unit = self.delgcn_layers[0]
        out = unit(A_list,Nodes_list,nodes_mask_list)[-1]
        # # # #three-layers
        # for unit in self.delgcn_layers:
        #     Nodes_list = unit(A_list,Nodes_list,nodes_mask_list)
        # out = Nodes_list[-1]

        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return out


class LightGCN(object):
    def __init__(self,n_layers,keep_prob) -> None:
        super().__init__()
        
        self.n_layers = n_layers
        self.keep_prob = keep_prob 

    def dropout(self, x, keep_prob):
        # 
        size = x.size() 
        index = x.coalesce().indices().t()
        values = x.coalesce().values() 
        
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()

        index = index[random_index]

        values = values[random_index]/keep_prob

        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def graph_diffusion(self,edge_index:torch.Tensor,feats:torch.Tensor,weight_mat):
        """
        propagate methods for lightGCN, 
        !!! edge_index must be a sparse tensor
        """       
        #   torch.split(all_emb , [self.n_users, self.n_items])
        all_emb = torch.mm(feats,weight_mat)
        embs = [all_emb]

        if self.keep_prob:

            g_droped = self.dropout(edge_index,self.keep_prob)
    
        else:
            g_droped = edge_index  

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        return light_out


class DELGCNLayer(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.args.out_feats,nhead=self.args.n_heads)
        self.evolve_weights = nn.TransformerEncoder(encoder_layer,num_layers=self.args.de_layers)
        self.lgcn = LightGCN(n_layers=self.args.lgcn_layers,keep_prob=self.args.keep_prob)
        self.activation = self.args.activation
        # self.GCN_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        # self.reset_param(self.GCN_weights)
        self.GCN_weights = torch.randn(self.args.in_feats,self.args.out_feats).to(args.device)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,node_embs_list,mask_list=None):

        # # step-by-step
        # out_seq = []
        # GCN_weights = self.GCN_weights.unsqueeze(0)
        # for Ahat,node_embs in zip(A_list,node_embs_list):
        #     # #one-step history data
        #     GCN_weights = self.evolve_weights(GCN_weights)
        #     node_embs = self.lgcn.graph_diffusion(Ahat,node_embs,GCN_weights[-1])
        #     out_seq.append(node_embs)
            # #all history data
            # temp_weights = self.evolve_weights(GCN_weights)
            
            # new_embs = self.lgcn.graph_diffusion(Ahat,node_embs,temp_weights[-1])

            # out_seq.append(new_embs)

            # GCN_weights = torch.cat([GCN_weights,temp_weights],dim=0)
        # # # one-shot 
        GCN_weights = torch.stack([self.GCN_weights] * len(A_list))
        
        GCN_weights = self.evolve_weights(GCN_weights)

        out_seq = [self.lgcn.graph_diffusion(Ahat,node_embs,weight) for Ahat,weight,node_embs in zip(A_list,GCN_weights,node_embs_list)]


        return out_seq


##-----v1 three-layer + 一次训练全部------------
# best valid measure: 0.1764
# final performance 0.13

## ----v1 one-layer + 一次全部训练--------------
### w0) ep 68 - Best valid measure:0.17380904923260915
#the test performance of current epoch --68-- is:0.1595144469140024

# ------v2 three-layer 逐步训练+一步数据---------
# ### w0) ep 83 - Best valid measure:0.18119551681195517
# the test performance of current epoch --83-- is:0.15649560795191864

## ---one-layer + 逐步训练+一步数据--------------
### w0) ep 10 - Best valid measure:0.17204301075268819
# the test performance of current epoch --10-- is:0.1655860349127182

## --- one-layer + 逐步训练 + 全部历史数据----
### w0) ep 33 - Best valid measure:0.17591763652641002
# the test performance of current epoch --33-- is:0.15827093260721578

## ---three-layer+逐步训练+全部历史数据----
### w0) ep 93 - Best valid measure:0.17941773865944485
# the test performance of current epoch --93-- is:0.15275096525096526

## ---one-layer+逐步训练+一次历史数据+固定gcn参数---------
### w0) ep 104 - Best valid measure:0.1744561705793668
# the test performance of current epoch --104-- is:0.15837937384898712
## ---one-layer+一次训练+固定gcn参数----
### w0) ep 35 - Best valid measure:0.17445054945054944
# the test performance of current epoch --35-- is:0.15962277331470487