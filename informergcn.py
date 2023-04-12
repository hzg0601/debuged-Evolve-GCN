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

class IGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False):
        super().__init__()
        IGB_args = u.Namespace({})
        # 规定每一层的特征的维度
        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.igb_layers = []
        self._parameters = nn.ParameterList()
        # 似乎只有3层
        for i in range(1,len(feats)):
            IGB_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            igb = InformerGCNBlock(IGB_args)
            #print (i,'igb', igb)
            self.igb_layers.append(igb.to(self.device))
            self._parameters.extend(list(self.igb_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list):
        node_feats= Nodes_list[-1]

        for unit in self.igb_layers:
            # GECU层是每次都初始化gcn，但保留上一次学习到的嵌入矩阵列表
            # 用上一次学习到的嵌入矩阵列表进行训练
            Nodes_list = unit(A_list,Nodes_list,nodes_mask_list)

        out = Nodes_list[-1]
        # 将学习的特征与原始特征拼接
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return out
        

class LightGCN(torch.nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.args = args
        # lgcn_args = u.Namespace({})

        self.in_features = args.feats_per_node
        self.emb_dim = args.out_feats 
        self.n_layers = args.n_layers
        self.keep_prob = args.keep_prob 

        self.weight = torch.nn.Linear(in_features=args.feats_per_node,out_features=args.out_feats)
        # random normal init seems to be a better choice when lightGCN 
        # actually don't use any non-linear activation function
        nn.init.normal_(self.weight, std=0.1)

    def dropout(self, x, keep_prob):
        # 
        size = x.size() 
        index = x.indices().t()
        values = x.values() 
        
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()

        index = index[random_index]

        values = values[random_index]/keep_prob

        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def graph_diffusion(self,edge_index:torch.Tensor,feats:torch.Tensor):
        """
        propagate methods for lightGCN, 
        !!! edge_index must be a sparse tensor
        """       
        #   torch.split(all_emb , [self.n_users, self.n_items])
        embs = [torch.mm(feats,self.weight)]

        if self.keep_prob:

            g_droped = self.dropout(edge_index,self.keep_prob)
    
        else:
            g_droped = edge_index  

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        return light_out


class InformerGCNBlock(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        informer_args = u.Namespace({})
        lgcn_args = u.Namespace({})

        self.evolve_weights = Informer(informer_args)
        self.lgcn = LightGCN(lgcn_args)
        self.activation = self.args.activation
        self.GCN_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,node_embs_list,mask_list=None):

        GCN_weights = self.GCN_weights
        out_seq = []
        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]

            GCN_weights = self.evolve_weights(GCN_weights) 

            node_embs = self.lgcn(Ahat,GCN_weights,node_embs)

            out_seq.append(node_embs)

        return out_seq
