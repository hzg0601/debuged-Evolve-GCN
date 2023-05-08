import torch
import utils as u
import numpy as np
import time

# edgelist中各列的位置
ECOLS = u.Namespace({'source': 0,
                     'target': 1,
                     'time': 2,
                     'label':3}) #--> added for edge_cls

def get_2_hot_deg_feats(adj,max_deg_out,max_deg_in,num_nodes):
    """
    以边的出度和入度的one-hot编码向量的concat为节点的度特征
    """
    #For now it'll just return a 2-hot vector
    #? 边的权重为1
    adj['vals'] = torch.ones(adj['idx'].size(0))
    degs_out, degs_in = get_degree_vects(adj,num_nodes)
    # idx: 第一列为从0开始的index，第二列为每个节点的度
    # vals: 1向量
    degs_out = {'idx': torch.cat([torch.arange(num_nodes).view(-1,1),
                                  degs_out.view(-1,1)],dim=1),
                'vals': torch.ones(num_nodes)}
    #* 构造一个num_nodes\times max_deg_out的矩阵，以出度的数目为行中1的位置，即one-hot编码度
    degs_out = u.make_sparse_tensor(degs_out,'long',[num_nodes,max_deg_out])
    
    #* 构造一个num_nodes\times max_deg_in的矩阵，以入度的数目为行中1的位置
    degs_in = {'idx': torch.cat([torch.arange(num_nodes).view(-1,1),
                                  degs_in.view(-1,1)],dim=1),
                'vals': torch.ones(num_nodes)}
    degs_in = u.make_sparse_tensor(degs_in,'long',[num_nodes,max_deg_in])
    # 合并出度向量和入度向量作为节点的特征，如
    hot_2 = torch.cat([degs_out,degs_in],dim = 1)
    hot_2 = {'idx': hot_2._indices().t(),
             'vals': hot_2._values()}

    return hot_2

def get_1_hot_deg_feats(adj,max_deg,num_nodes):
    """
    以边的出度的one-hot编码为节点的度特征
    """
    #For now it'll just return a 2-hot vector
    new_vals = torch.ones(adj['idx'].size(0))
    new_adj = {'idx':adj['idx'], 'vals': new_vals}
    degs_out, _ = get_degree_vects(new_adj,num_nodes)
    
    degs_out = {'idx': torch.cat([torch.arange(num_nodes).view(-1,1),
                                  degs_out.view(-1,1)],dim=1),
                'vals': torch.ones(num_nodes)}
    
    # print ('XXX degs_out',degs_out['idx'].size(),degs_out['vals'].size())
    degs_out = u.make_sparse_tensor(degs_out,'long',[num_nodes,max_deg])

    hot_1 = {'idx': degs_out._indices().t(),
             'vals': degs_out._values()}
    return hot_1

def get_max_degs(args,dataset,all_window=False):
    """
    计算邻接矩阵的最大出度和最大入度，
    如果关键词all_window为True,则使用所有时间窗口，即计算每个[0,t](t\in [min,max])内的邻接矩阵；
    如果为False,则计算args条件给定的[t-adj_mat_time_window,t](t\in [min,max])时间窗口内的邻接矩阵
    即如果all_window为True是起点固定的，如果all_window为False是窗口长度固定的。
    """
    max_deg_out = []
    max_deg_in = []
    for t in range(dataset.min_time, dataset.max_time):
        if all_window:
            window = t+1
        else:
            window = args.adj_mat_time_window
        #? t和window写反了?如果没有则计算的是每个[0,t](t\in [min,max])内的邻接矩阵
        #? 如果
        cur_adj = get_sp_adj(edges = dataset.edges,
                             time = t,
                             weighted = False,
                             time_window = window)
        # print(window)
        cur_out, cur_in = get_degree_vects(cur_adj,dataset.num_nodes)
        max_deg_out.append(cur_out.max())
        max_deg_in.append(cur_in.max())
        # max_deg_out = torch.stack([max_deg_out,cur_out.max()]).max()
        # max_deg_in = torch.stack([max_deg_in,cur_in.max()]).max()
    # exit()
    max_deg_out = torch.stack(max_deg_out).max()
    max_deg_in = torch.stack(max_deg_in).max()
    max_deg_out = int(max_deg_out) + 1
    max_deg_in = int(max_deg_in) + 1
    
    return max_deg_out, max_deg_in

def get_max_degs_static(num_nodes, adj_matrix):
    """
    计算静态图的最大出度和入度
    """
    cur_out, cur_in = get_degree_vects(adj_matrix, num_nodes)
    max_deg_out = int(cur_out.max().item()) + 1
    max_deg_in = int(cur_in.max().item()) + 1
    
    return max_deg_out, max_deg_in


def get_degree_vects(adj,num_nodes):
    """
    获取每个节点的出度和入度
    """
    adj = u.make_sparse_tensor(adj,'long',[num_nodes])
    degs_out = adj.matmul(torch.ones(num_nodes,1,dtype = torch.long))
    degs_in = adj.t().matmul(torch.ones(num_nodes,1,dtype = torch.long))
    return degs_out, degs_in

def get_sp_adj(edges,time,weighted,time_window):
    """
    构造sparse的邻接矩阵，选择交易时间在[time-time_window,time]内的边构成邻接矩阵
    
    edges: 边列表
    time: 时间点，构造中只保留时间小于该时间点的节点
    weighted: 如果为True,则使用value进行加权，否额权重为1；
    time_window: 时间窗口，即构造边的时间段为[time-time_window,time]
    """
    idx = edges['idx'] #* edges['idx']包括source,target,time,label四列
    subset = idx[:,ECOLS.time] <= time # 选择交易时间点小于time的边
    #* 选择交易时间大于time-time_window的边，即在时间点time内给定时间窗口的交易
    subset = subset * (idx[:,ECOLS.time] > (time - time_window)) #
    idx = edges['idx'][subset][:,[ECOLS.source, ECOLS.target]]  
    vals = edges['vals'][subset]
    out = torch.sparse.FloatTensor(idx.t(),vals).coalesce()
    
    idx = out._indices().t()
    if weighted:
        vals = out._values()
    else:
        vals = torch.ones(idx.size(0),dtype=torch.long)

    return {'idx': idx, 'vals': vals}


def get_edge_labels(edges,time):
    """
    选择等于给定时间点的边列表的标签
    """
    idx = edges['idx']
    subset = idx[:,ECOLS.time] == time
    idx = edges['idx'][subset][:,[ECOLS.source, ECOLS.target]]  
    vals = edges['idx'][subset][:,ECOLS.label]

    return {'idx': idx, 'vals': vals}

# 指定一个最大节点数，每个邻接矩阵由于只能包含一部分节点，因此需要对不包含的节点进行掩码，
# 指定该邻接矩阵不包含的索引的值为负无穷大。
def get_node_mask(cur_adj,num_nodes):
    """
    掩码每个时间步内邻接矩阵不包含的节点
    """
    mask = torch.zeros(num_nodes) - float("Inf")
    non_zero = cur_adj['idx'].unique()

    mask[non_zero] = 0
    
    return mask


def get_static_sp_adj(edges,weighted):
    """
    构造静态稀疏邻接矩阵
    """
    idx = edges['idx'][:,:2] #* edges['idx']包括source,target,time,但构造静态邻接矩阵仅需要前三列

    if weighted:
        vals = edges['vals']# * [subset]
    else:
        vals = torch.ones(idx.size(0),dtype = torch.long)

    return {'idx': idx, 'vals': vals}

def get_sp_adj_only_new(edges,time,weighted):
    return get_sp_adj(edges, time, weighted, time_window=1)

# gcn标准化
def normalize_adj(adj,num_nodes):
    '''
    执行gcn标准化
    takes an adj matrix as a dict with idx and vals and normalize it by: 
        - adding an identity matrix, 
        - computing the degree vector
        - multiplying each element of the adj matrix (aij) by (di*dj)^-1/2
    '''
    idx = adj['idx']
    vals = adj['vals']

    
    sp_tensor = torch.sparse.FloatTensor(idx.t(),vals.type(torch.float),torch.Size([num_nodes,num_nodes]))
    
    sparse_eye = make_sparse_eye(num_nodes)
    sp_tensor = sparse_eye + sp_tensor

    idx = sp_tensor._indices()
    vals = sp_tensor._values()

    degree = torch.sparse.sum(sp_tensor,dim=1).to_dense()
    di = degree[idx[0]]
    dj = degree[idx[1]]

    vals = vals * ((di * dj) ** -0.5)
    
    return {'idx': idx.t(), 'vals': vals}

def make_sparse_eye(size):
    """
    构造单位阵的稀疏矩阵表示
    """
    eye_idx = torch.arange(size)
    eye_idx = torch.stack([eye_idx,eye_idx],dim=1).t()
    vals = torch.ones(size)
    eye = torch.sparse.FloatTensor(eye_idx,vals,torch.Size([size,size]))
    return eye

def get_all_non_existing_edges(adj,tot_nodes):
    """
    对全连接边列表中的边进行编码，将不在给定的adj的边全部取出
    adj->dict: 给定的邻接矩阵的字典,idx的值为边列表
    tot_nodes-> int: 总节点数
    """
    true_ids = adj['idx'].t().numpy()
    # 对给定的adj的边进行编码，它必为all_edges_ids的子集
    true_ids = get_edges_ids(true_ids,tot_nodes)

    all_edges_idx = np.arange(tot_nodes)
    #* 将a的每个点b的每个点组合,即全连接图的边列表
    all_edges_idx = np.array(np.meshgrid(all_edges_idx,
                                         all_edges_idx)).reshape(2,-1)
    # 对边进行编码
    all_edges_ids = get_edges_ids(all_edges_idx,tot_nodes)

    #only edges that are not in the true_ids should keep here
    #* 对给定adj中不在all_edges_ids中的边进行掩码
    #* np.logical_or,np.logical_and,对两个对象执行按位或、与
    #* np.logical_not 对对象执行按位非
    # 位运算符仅在对象维bool值时与逻辑运算符一致
    mask = np.logical_not(np.isin(all_edges_ids,true_ids))

    non_existing_edges_idx = all_edges_idx[:,mask]
    edges = torch.tensor(non_existing_edges_idx).t()
    vals = torch.zeros(edges.size(0), dtype = torch.long)
    return {'idx': edges, 'vals': vals}


def get_non_existing_edges(adj,number, tot_nodes, smart_sampling, existing_nodes=None):
    """
    边的负样本采样，如果定义smart_sampling为True，
    则只从adj['idx'][0]中采样source节点
    从existing_nodes采样target节点
    若定义smart_sampling为False,则从全部节点中随机采样
    采样的节点数小于等于number
    adj-> dict: 给定的邻接矩阵的字典,idx的值为边列表
    number->int: 控制边的最小采样数
    tot_nodes->int: 总节点数
    smart_sampling-> bool: 是否进行智能采样，即采样target节点时，只考虑在existing_nodes中的节点
    existing_nodes->np.array: target节点的候选集
    """
    # print('----------')
    t0 = time.time()
    idx = adj['idx'].t().numpy() # 2\times n，设边的数量为n
    true_ids = get_edges_ids(idx,tot_nodes)

    true_ids = set(true_ids)

    #the maximum of edges would be all edges that don't exist between nodes that have edges
    #? 边的最大值是那些不存在于有边的节点间的所有边,考虑了出、入两个方向？
    # idx.shape[1] * (idx.shape[1]-1) - len(true_ids)=n\times (n-1)- n=n^2-2n
    # 取number,n^2-2n二者中的最小值
    #* 边数小于等于n^2-2n
    num_edges = min(number,idx.shape[1] * (idx.shape[1]-1) - len(true_ids))
    #* 如果进行智能采样，则以existing_nodes为target候选集,以边列表第一列为source候选集
    #* 如果不进行智能采样，则全体节点为target、source候选集
    if smart_sampling:
        #existing_nodes = existing_nodes.numpy()
        def sample_edges(num_edges):
            # print('smart_sampling')
            from_id = np.random.choice(idx[0],size = num_edges,replace = True)
            to_id = np.random.choice(existing_nodes,size = num_edges, replace = True)
            #print ('smart_sampling', from_id, to_id)
            # 合并from_id,to_id构成边列表
            if num_edges>1:
                edges = np.stack([from_id,to_id])
            else:
                edges = np.concatenate([from_id,to_id])
            return edges
    else:
        def sample_edges(num_edges):
            # 如果num_edges>1，则从[0,tot_nodes]中采样num_edges对节点即num_edges条边
            # 如果num_edges<=1，则从[0,tot_nodes]采样一对节点，即一条边
            if num_edges > 1:
                edges = np.random.randint(0,tot_nodes,(2,num_edges))
            else:
                edges = np.random.randint(0,tot_nodes,(2,))
            return edges
    # 采样4倍num_edges条边
    edges = sample_edges(num_edges*4)
    # 编码采样的边
    edge_ids = edges[0] * tot_nodes + edges[1]
    # 从随机采样的4倍num_edges条边中，去除已采样、自环、真边，
    # 共采样num_edges条边
    out_ids = set() # 记录已采样的边的id
    num_sampled = 0 # 记录已采样的边数
    sampled_indices = [] # 记录已采样的边的顺序
    for i in range(num_edges*4):
        eid = edge_ids[i]
        #ignore if any of these conditions happen
        # 采样边时，已采样、自环、真边不采样
        if eid in out_ids or edges[0,i] == edges[1,i] or eid in true_ids:
            continue

        #add the eid and the index to a list
        out_ids.add(eid)
        sampled_indices.append(i)
        num_sampled += 1

        #if we have sampled enough edges break
        if num_sampled >= num_edges:
            break
    # 根据纪律的采样的边的顺序，选出边        
    edges = edges[:,sampled_indices]
    edges = torch.tensor(edges).t()
    vals = torch.zeros(edges.size(0),dtype = torch.long)
    return {'idx': edges, 'vals': vals}

def get_edges_ids(sp_idx, tot_nodes):
    """
    对边进行编码，方式是源节点的ID*总节点数+目标节点的ID
    sp_idx: 边列表，n\times 2，sp_idx[0]是源节点的id
    tot_nodes: 总节点数
    输出：n维向量
    """
    # print(sp_idx)
    # print(tot_nodes)
    # print(sp_idx[0]*tot_nodes)
    return sp_idx[0]*tot_nodes + sp_idx[1]


