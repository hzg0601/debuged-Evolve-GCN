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

class Informer(torch.nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})

class LightGCN(torch.nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.args = args
        lgcn_args = u.Namespace({})
class LightGCN(torch.nn.Module):
    def __init__(self, 
                 n_samples: int,
                 emb_dim:int = 128,
                 n_layers:int = 3,
                 keep_prob: Union[float,bool] = 0.95,# drop_out
                 loss_type:str = "bpr",
                 reg_loss: bool = False
                 ):
        super().__init__()
        self.n_samples = n_samples
        self.emb_dim = emb_dim # 嵌入维度
        self.n_layers = n_layers
        self.keep_prob = keep_prob # dropout的概率
        self.loss_type = loss_type 
        self.reg_loss = reg_loss
        self.embedding = torch.nn.Embedding(num_embeddings=self.n_samples, embedding_dim=self.emb_dim)

        # random normal init seems to be a better choice when lightGCN 
        # actually don't use any non-linear activation function
        nn.init.normal_(self.embedding.weight, std=0.1)

       
    # 如果定义的dropout参数，则按照给定的int(keep_prob+随机数)
    def __dropout_x(self, x, keep_prob):
        # 
        size = x.size() # 边列表的size
        index = x.indices().t()
        values = x.values() # 边
        # torch.rand(len(values))随机抽取len(values)个0-1均匀分布随机数，
        # 然后加上keep_prob， 然后取整、取bool，构成一个bool索引向量
        
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        # 按照bool索引向量取索引
        index = index[random_index]
        # 取出后对value进行重新归一化，
        values = values[random_index]/keep_prob
        # 将g转换为稀疏矩阵
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    # 如果A是分块矩阵(稀疏邻接矩阵的向量)，则按块进行dropout
    def __dropout(self, edge_index, keep_prob):
        """执行dropout_edge"""
        if self.A_split:
            graph = []
            for g in edge_index:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(edge_index, keep_prob)
        return graph
    
    def graph_diffusion(self,edge_index:torch.Tensor):
        """
        propagate methods for lightGCN, 
        !!! edge_index must be a sparse tensor
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.n_users, self.n_items])
        embs = [all_emb]
        # 训练中设置dropput参数才执行图的dropout，否则保留完整的图；
        if self.keep_prob:

            g_droped = self.__dropout(edge_index,self.keep_prob)
    
        else:
            g_droped = edge_index  
        # G * E构成一次传播,此处G为传播矩阵，在文中为对称标准化邻接矩阵。
        for layer in range(self.n_layers):
            # 如果是分块矩阵，则针对每个矩阵进行传播
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        # 将中间结果拼接，取均值构成最终输出，
        # 注意torch.stack相当于构造一个列表，维度会增加
        # torch.cat才会保持维度
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        # torch.split，切割张量,第二个参数为切分后每块的大小，默认在第0维
        users_out, items_out = torch.split(light_out, [self.n_users, self.n_items])
        
        return users_out, items_out
    
    def cal_reg_loss(self,users,pos,neg):
        """
        计算正则损失
        """
        length = len(users)
        weight = 0
        for idx,emb_layer in zip([users, pos, neg],
                                 [self.embedding_user, self.embedding_item, self.embedding_item]):
            if idx is not None:
                weight += emb_layer(idx).norm(2).pow(2)
        reg_loss = weight/length/2
        return reg_loss
    # 计算BPR损失
    def bpr_loss(self, users_emb, pos_emb, neg_emb): 
        """计算bpr损失"""
        pos_scores = torch.mul(users_emb, pos_emb).sum(-1,keepdim=True)
        if neg_emb is not None:
            users_emb = users_emb.unsqueeze(1) 
            neg_scores = torch.mul(users_emb, neg_emb).sum(-1)
        else:
            neg_scores = 0 
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        return loss

    def cosine_loss(self, users_emb, items_emb,neg_emb):
        "基于边的cosine损失"
        users_emb = F.normalize(users_emb,p=2,dim=-1)
        items_emb = F.normalize(items_emb, p=2, dim=-1)
        
        pos_inner = torch.mul(users_emb, items_emb).sum(-1,keepdim=True)
        
        loss = torch.mean(1-pos_inner)
        
        if neg_emb is not None:
            users_emb = users_emb.unsqueeze(1) 
            neg_emb = F.normalize(neg_emb, p=2, dim=-1)
            neg_inner = torch.mul(users_emb, neg_emb).sum(-1)
            loss += torch.mean(neg_inner)
        
        return loss 

    def forward(self, edge_index:torch.Tensor, users=None, pos=None, neg=None):
        """
        !!! edge_index must be a sparse tensor
        calculate loss 
        """
        user_out, item_out = self.graph_diffusion(edge_index)
        self._users_emb, self._items_emb = user_out.detach(), item_out.detach()

        edge_index = edge_index.indices()
            
        if users is None:
            users = edge_index[0]
            
        if pos is None:
            pos = edge_index[1] -self.n_users if edge_index[1].min()>=self.n_users else edge_index[1]
            
        if neg is not None:
            # 如果负采样没有区分类型，则拼接起来抽取
            try:
                neg = neg - self.n_users if neg.min() >= self.n_users else neg 
                neg_emb = item_out[neg]   
            except Exception as e:
                #logger.warning(e)
                neg_emb = torch.cat([user_out, item_out],dim=0)[neg] 
        else:
            neg_emb = None 
        users_emb = user_out[users]
        pos_emb = item_out[pos] 
        
        if self.loss_type == "bpr":
            loss = self.bpr_loss(users_emb, pos_emb, neg_emb)
        elif self.loss_type == "cosine":
            loss = self.cosine_loss(users_emb, pos_emb, neg_emb)
        if self.reg_loss:
            loss += self.cal_reg_loss(users, pos, neg)    
        return loss 
    
    def get_embedding(self, idx=None, tar_type="user"):
        """获取目标idx的嵌入向量"""
        if tar_type == "item":
            idx = idx - self.n_users if idx.min() >= self.n_users else idx 
            embs = self._items_emb[idx] if idx is not None else self._items_emb
        elif tar_type == "user":
            embs = self._users_emb[idx] if idx is not None else self._users_emb
        else:
            raise ValueError
        return embs 
    # 计算给定users对全体item的排序
    def get_rating_mat(self, users=None):
        all_users, all_items = self._users_emb, self._items_emb
        users_emb = all_users[users.long()] if users is not None else all_users # 
        items_emb = all_items
        rating = torch.softmax(torch.matmul(users_emb, items_emb.t()),dim=1)
        return rating
    
    def get_sim_mat(self,idx=None,tar_type="user"):
        """calculate similarity matrix"""

        embs = self.get_embedding(idx,tar_type=tar_type)
        embs = F.normalize(embs, p=2, dim=1) 
        sim_mat = torch.mm(embs, embs.t())

        return sim_mat

class InformerGCNBlock(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        informer_args = u.Namespace({})
        lgcn_args = u.Namespace({})
        # cell_args，GRU的参数，仅仅是参数矩阵的大小
        self.evolve_weights = Informer(informer_args)
        self.lgcn = LightGCN(lgcn_args)
        self.activation = self.args.activation
        self.GCN_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)
    # 对EvolveGCN的每一层，训练全部历史数据
    # 得到一个历史GCN模型的列表
    # 根据每个GCN模型的参数矩阵，得到一个嵌入矩阵列表
    def forward(self,A_list,node_embs_list,mask_list=None):
        #?每次都初始化GCN列表？？？
        GCN_weights = self.GCN_weights
        out_seq = []
        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            #first evolve the weights from the initial and use the new weights with the node_embs
            # 是GCN的权重矩阵,rows_g[i]\times cols_g[j]
            GCN_weights = self.evolve_weights(GCN_weights) # mat_GRU_cell
            # GCN嵌入表示层
            # ! 因此该模型的结构是一层GCN一层RNN
            # !每一层EvolveGCN的单元只考虑一层的GCN的参数
            # num\times rows_g[i] matmul rows_g[i]\times cols_g[i] => num\times cols_g[i]
            # 
            node_embs = self.lgcn(Ahat,GCN_weights,node_embs)

            out_seq.append(node_embs)

        return out_seq
