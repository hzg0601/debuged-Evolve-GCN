import torch
import taskers_utils as tu
import utils as u

"""
链路预测任务的数据准备类，主要为：
时间点、历史节点的特征列表、历史标签列表、历史邻接矩阵列表、历史掩码列表
注意：用于节点分类和链路预测的数据集不通用，链接预测任务数据集自定义了
prepare_node_feats方法
"""
class Link_Pred_Tasker():
	'''
	Creates a tasker object which computes the required inputs for training on a link prediction
	task. It receives a dataset object which should have two attributes: nodes_feats and edges, this
	makes the tasker independent of the dataset being used (as long as mentioned attributes have the same
	structure).

	Based on the dataset it implements the get_sample function required by edge_cls_trainer.
	This is a dictionary with:
		- time_step: the time_step of the prediction
		- hist_adj_list: the input adjacency matrices until t, each element of the list 
						 is a sparse tensor with the current edges. For link_pred they're
						 unweighted
		- nodes_feats_list: the input nodes for the GCN models, each element of the list is a tensor
						  two dimmensions: node_idx and node_feats
		- label_adj: a sparse representation of the target edges. A dict with two keys: idx: M by 2 
					 matrix with the indices of the nodes conforming each edge, vals: 1 if the node exists
					 , 0 if it doesn't

	There's a test difference in the behavior, on test (or development), the number of sampled non existing 
	edges should be higher.
	'''
	def __init__(self,args,dataset):
		self.data = dataset
		#max_time for link pred should be one before
		self.max_time = dataset.max_time - 1 #? 链路预测任务的最大时间为数据集的最大时间-1
		self.args = args
		self.num_classes = 2 # 链路预测任务是一种分类任务，只有链接和不链接两类
		# 如果不使用节点的度one-hot向量，每个节点特征的维度即数据的固有维度
		# 如果使用节点的度one-hot向量，则调用build_get_node_feats，
		# 根据出度、入度定义特征向量的维度
		if not (args.use_2_hot_node_feats or args.use_1_hot_node_feats):
			self.feats_per_node = dataset.feats_per_node

		self.get_node_feats = self.build_get_node_feats(args,dataset)
		self.prepare_node_feats = self.build_prepare_node_feats(args,dataset)
		self.is_static = False
		
		'''TO CREATE THE CSV DATASET TO USE IN DynGEM
		print ('min max time:', self.data.min_time, self.data.max_time)
		file = open('data/autonomous_syst100_adj.csv','w')
		file.write ('source,target,weight,time\n')
		for time in range(self.data.min_time, self.data.max_time):
			adj_t = tu.get_sp_adj(edges = self.data.edges,
					   time = time,
					   weighted = True,
					   time_window = 1)
			#node_feats = self.get_node_feats(adj_t)
			print (time, len(adj_t))
			idx = adj_t['idx']
			vals = adj_t['vals']
			num_nodes = self.data.num_nodes
			sp_tensor = torch.sparse.FloatTensor(idx.t(),vals.type(torch.float),torch.Size([num_nodes,num_nodes]))
			dense_tensor = sp_tensor.to_dense()
			idx = sp_tensor._indices()
			for i in range(idx.size()[1]):
				i0=idx[0,i]
				i1=idx[1,i]
				w = dense_tensor[i0,i1]
				file.write(str(i0.item())+','+str(i1.item())+','+str(w.item())+','+str(time)+'\n')

			#for i, v in zip(idx, vals):
			#	file.write(str(i[0].item())+','+str(i[1].item())+','+str(v.item())+','+str(time)+'\n')

		file.close()
		exit'''

	# def build_get_non_existing(args):
	# 	if args.use_smart_neg_sampling:
	# 	else:
	# 		return tu.get_non_existing_edges

	def build_prepare_node_feats(self,args,dataset):
		"""
		如果使用使用节点的度one-hot向量，则返回one-hot编码的度特征作为特征向量；
		如果不使用节点的度one-hot向量，则直接调用数据的prepare_node_feats方法，返回节点的固有特征；
		并组装成torch.tensor
  		"""
		if args.use_2_hot_node_feats or args.use_1_hot_node_feats:
			def prepare_node_feats(node_feats):
				return u.sparse_prepare_tensor(node_feats,
											   torch_size= [dataset.num_nodes,
											   				self.feats_per_node])
		else:
			prepare_node_feats = self.data.prepare_node_feats 

		return prepare_node_feats


	def build_get_node_feats(self,args,dataset):
		"""
		根据预定义的特征形式，设定节点特征的维度，抽取numpy.ndarray形式的特征
		输出作为prepare_node_feats的输入

  		"""
		if args.use_2_hot_node_feats:
			max_deg_out, max_deg_in = tu.get_max_degs(args,dataset)
			self.feats_per_node = max_deg_out + max_deg_in
			def get_node_feats(adj):
				return tu.get_2_hot_deg_feats(adj,
											  max_deg_out,
											  max_deg_in,
											  dataset.num_nodes)
		elif args.use_1_hot_node_feats:
			max_deg,_ = tu.get_max_degs(args,dataset)
			self.feats_per_node = max_deg
			def get_node_feats(adj):
				return tu.get_1_hot_deg_feats(adj,
											  max_deg,
											  dataset.num_nodes)
		else:
			def get_node_feats(adj):
				return dataset.nodes_feats

		return get_node_feats


	def get_sample(self,idx,test, **kwargs):
		"""
		获取样本的历史邻接矩阵列表、节点特征列表、节点掩码列表、标签，
		该类的主方法
		idx: 指定时间点，时间区间为[idx-num_hist_steps,idx+1)
  		"""
		hist_adj_list = []
		hist_ndFeats_list = []
		hist_mask_list = []
		existing_nodes = []
		for i in range(idx - self.args.num_hist_steps, idx+1):
			# 返回值为dict,idx为节点的边列表；vals为边的权重值
			# 构造sparse的邻接矩阵，选择交易时间在[time-time_window,time]内的边构成邻接矩阵
			cur_adj = tu.get_sp_adj(edges = self.data.edges, 
								   time = i,
								   weighted = True,
								   time_window = self.args.adj_mat_time_window)
			#* 如果进行智能采样，则以existing_nodes为target候选集,以边列表第一列为source候选集
    		#* 如果不进行智能采样，则全体节点为target、source候选集
			#* torch.unique(),将Tensor中的所有点构成一个向量，对向量去重，返回的是一个1维Tensor
			if self.args.smart_neg_sampling:
				existing_nodes.append(cur_adj['idx'].unique())
			else:
				existing_nodes = None

			node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)

			node_feats = self.get_node_feats(cur_adj)

			cur_adj = tu.normalize_adj(adj = cur_adj, num_nodes = self.data.num_nodes)

			hist_adj_list.append(cur_adj)
			hist_ndFeats_list.append(node_feats)
			hist_mask_list.append(node_mask)

		# This would be if we were training on all the edges in the time_window
		#* 链路预测中的标签即邻接矩阵，因此以下一个时间点的邻接矩阵为标签邻接矩阵
		label_adj = tu.get_sp_adj(edges = self.data.edges, 
								  time = idx+1,
								  weighted = False,
								  time_window =  self.args.adj_mat_time_window)
		if test:
			neg_mult = self.args.negative_mult_test # 采样的个数
		else:
			neg_mult = self.args.negative_mult_training
			
		if self.args.smart_neg_sampling:
			existing_nodes = torch.cat(existing_nodes)

		# 如果all_edges为True,对全连接边列表中的边进行编码，将不在给定的adj的边全部取出
		if 'all_edges' in kwargs.keys() and kwargs['all_edges'] == True:
			non_exisiting_adj = tu.get_all_non_existing_edges(adj = label_adj, tot_nodes = self.data.num_nodes)
		# 如果all_edges为False, 则从当前时间点的邻接矩阵中选择目标节点，从label_adj[0]中采样源节点
		# 采样number条边
		else:
			non_exisiting_adj = tu.get_non_existing_edges(adj = label_adj,
													  number = label_adj['vals'].size(0) * neg_mult,
													  tot_nodes = self.data.num_nodes,
													  smart_sampling = self.args.smart_neg_sampling,
													  existing_nodes = existing_nodes)

		# label_adj = tu.get_sp_adj_only_new(edges = self.data.edges,
		# 								   weighted = False,
		# 								   time = idx)
		
		label_adj['idx'] = torch.cat([label_adj['idx'],non_exisiting_adj['idx']])
		label_adj['vals'] = torch.cat([label_adj['vals'],non_exisiting_adj['vals']])
		return {'idx': idx,
				'hist_adj_list': hist_adj_list,
				'hist_ndFeats_list': hist_ndFeats_list,
				'label_sp': label_adj,
				'node_mask_list': hist_mask_list}

