import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
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


"""

class EGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False):
        super().__init__()
        GRCU_args = u.Namespace({})
        # 规定每一层的特征的维度
        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        # 似乎只有3层
        for i in range(1,len(feats)):
            GRCU_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            grcu_i = GRCU(GRCU_args)
            #print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list):
        node_feats= Nodes_list[-1]

        for unit in self.GRCU_layers:
            # GECU层是每次都初始化gcn，但保留上一次学习到的嵌入矩阵列表
            # 用上一次学习到的嵌入矩阵列表进行训练
            Nodes_list = unit(A_list,Nodes_list,nodes_mask_list)

        out = Nodes_list[-1]
        # 将学习的特征与原始特征拼接
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return out


class GRCU(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats
        # cell_args，GRU的参数，仅仅是参数矩阵的大小
        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)
    # 对EvolveGCN的每一层，训练全部历史数据
    # 得到一个历史GCN模型的列表
    # 根据每个GCN模型的参数矩阵，得到一个嵌入矩阵列表
    def forward(self,A_list,node_embs_list,mask_list):
        #?每次都初始化GCN列表？？？
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            #first evolve the weights from the initial and use the new weights with the node_embs
            # 是GCN的权重矩阵,rows_g[i]\times cols_g[j]
            GCN_weights = self.evolve_weights(GCN_weights,node_embs,mask_list[t])
            # GCN嵌入表示层
            # ! 因此该模型的结构是一层GCN一层RNN
            # !每一层EvolveGCN的单元只考虑一层的GCN的参数
            # num\times rows_g[i] matmul rows_g[i]\times cols_g[i] => num\times cols_g[i]
            # 
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq
# mat_GRU输出的是GCN的权重矩阵，是GCN的rows\times cols 
class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        # TopK可以将输入的X_t{num\times rows}转换为rows\times k的矩阵,
        # 即rows\times cols的矩阵
        # 执行mat_GRU_gate，得到rows\times cols
        self.choose_topk = TopK(feats = args.rows,
                                k = args.cols)
    # prev_Q,相当于论文中的H_{t-1}；prev_Z相当于X_t，
    # 论文中似乎存在一些符号滥用的问题，summary中Z_t用\tilde{X}_t更合适
    # 得到的是一个rows\times cols的矩阵，完成参数更新后可以利用gcn进行嵌入得到更新的prev_Z
    def forward(self,prev_Q,prev_Z,mask):
        z_topk = self.choose_topk(prev_Z,mask)
        # update相当于论文中的Z_t
        # mat_GRU_gate执行W* key_1, U* key_2,论文中W,U,B都是参数
        # 在H版本2中，将z_topk视为X_t
        update = self.update(z_topk,prev_Q)
        # reset相当于论文中的R_t
        reset = self.reset(z_topk,prev_Q)
        # 更新\tilde{H}_t
        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

        
# 执行\sigmoid(WX+UH+B)操作，在GRU和LSTM中，该操作为2,3步的通用操作
class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1)) # y=Xp/||p||中的p
        # 用于计算行权重的权重向量，其长度为rows
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)
    # 论文中没有关于mask的信息，由tasker_utils.py的代码可推测，
    # 模型设定了一个最大节点参数，由于每个邻接矩阵必然只包含其中一部分节点
    # 因此有些节点不能被选择，设定这些节点的权重为负无穷大，以保证它们在排序中
    # 不会被选中，对于任何版本的模型都需如此
    
    def forward(self,node_embs,mask):
        # node_embs.shape matmul rows\times 1,得到rows\times 1
        # 故node_embs的shape必然为dim\times rows
        
        # 由于得到的scores为rows \times 1，必然有rows=num_nodes
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        # mask为num_nodes \times 1的向量
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices,self.k)
            
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()
            # 由于node_embs的shape为dim\times rows
        # 选择的是前k行得到的是k\times rows, 另一个是k\times 1
        # 执行广播机制,得到一个 k\times rows的矩阵
        # 最终进行转置得到rows\times k的矩阵

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1,1))

        #we need to transpose the output
        return out.t()
