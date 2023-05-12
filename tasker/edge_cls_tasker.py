import torch
import taskers_utils as tu
import utils as u


class Edge_Cls_Tasker():
	def __init__(self,args,dataset):
		self.data = dataset
		#max_time for link pred should be one before
		self.max_time = dataset.max_time
		self.args = args
		self.num_classes = dataset.num_classes

		if not args.use_1_hot_node_feats:
			self.feats_per_node = dataset.feats_per_node

		self.get_node_feats = self.build_get_node_feats(args,dataset)
		self.prepare_node_feats = self.build_prepare_node_feats(args,dataset)
		
		self.is_static = False

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


	def get_sample(self,idx,test):
		"""
		获取样本的历史邻接矩阵列表、节点特征列表、节点掩码列表、标签，
		该类的主方法
		idx: 指定时间点，时间区间为[idx-num_hist_steps,idx+1)
		"""
		hist_adj_list = []
		hist_ndFeats_list = []
		hist_mask_list = []

		for i in range(idx - self.args.num_hist_steps, idx+1):
			cur_adj = tu.get_sp_adj(edges = self.data.edges, 
								   time = i,
								   weighted = True,
								   time_window = self.args.adj_mat_time_window)
			node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)
			node_feats = self.get_node_feats(cur_adj)
			cur_adj = tu.normalize_adj(adj = cur_adj, num_nodes = self.data.num_nodes)

			hist_adj_list.append(cur_adj)
			hist_ndFeats_list.append(node_feats)
			hist_mask_list.append(node_mask)
		# 获取给定时间点的边的标签，返回值是一个dict,边列表和边标签
		label_adj = tu.get_edge_labels(edges = self.data.edges, 
								  	   time = idx)

		return {'idx': idx,
				'hist_adj_list': hist_adj_list,
				'hist_ndFeats_list': hist_ndFeats_list,
				'label_sp': label_adj,
				'node_mask_list': hist_mask_list}

