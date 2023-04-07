import utils as u
import torch
import torch.distributed as dist
import numpy as np
import time
import random

#datasets
import bitcoin_dl as bc
import elliptic_temporal_dl as ell_temp
import uc_irv_mess_dl as ucim
import auto_syst_dl as aus
import sbm_dl as sbm
import reddit_dl as rdt


#taskers
import link_pred_tasker as lpt
import edge_cls_tasker as ect
import node_cls_tasker as nct

#models
import models as mls
import egcn_h
import egcn_o


import splitter as sp
import Cross_Entropy as ce

import trainer as tr

import logger


def random_param_value(param, param_min, param_max, type='int'):
	"""
	对参数进行随机初始化
	param: 待初始化参数
	param_min->int: 参数初始化最小值
	param_max->int: 参数初始化最大值
	type->str: 参数值类型
	"""
	if str(param) is None or str(param).lower()=='none':
		if type=='int':
			return random.randrange(param_min, param_max+1)
		elif type=='logscale':
			#? 相当于对取出的区间内的数取exp
			interval=np.logspace(np.log10(param_min), np.log10(param_max), num=100)
			return np.random.choice(interval,1)[0]
		else:
			return random.uniform(param_min, param_max)
	else:
		return param

def build_random_hyper_params(args):
	"""
	设定模型执行顺序、学习率、模型每层的维度
	"""
	if args.model == 'all':
		model_types = ['gcn', 'egcn_o', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank] # rank,各模型执行的顺序
	elif args.model == 'all_nogcn':
		model_types = ['egcn_o', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
	elif args.model == 'all_noegcn3':
		model_types = ['gcn', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
	elif args.model == 'all_nogruA':
		model_types = ['gcn', 'egcn_o', 'egcn_h', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
		args.model=model_types[args.rank]
	elif args.model == 'saveembs':
		model_types = ['gcn', 'gcn', 'skipgcn', 'skipgcn']
		args.model=model_types[args.rank]

	args.learning_rate =random_param_value(args.learning_rate, args.learning_rate_min, args.learning_rate_max, type='logscale')
	# args.adj_mat_time_window = random_param_value(args.adj_mat_time_window, args.adj_mat_time_window_min, args.adj_mat_time_window_max, type='int')
	# 如果模型为gcn，令num_hist_steps=0；如果模型为其他模型，则在[num_hist_steps_min,num_hist_steps_max]中随机选择一个
	if args.model == 'gcn':
		args.num_hist_steps = 0
	else:
		args.num_hist_steps = random_param_value(args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max, type='int')
	# 设定模型每层的维度
	args.gcn_parameters['feats_per_node'] =random_param_value(args.gcn_parameters['feats_per_node'], args.gcn_parameters['feats_per_node_min'], args.gcn_parameters['feats_per_node_max'], type='int')
	args.gcn_parameters['layer_1_feats'] =random_param_value(args.gcn_parameters['layer_1_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
	if args.gcn_parameters['layer_2_feats_same_as_l1'] or args.gcn_parameters['layer_2_feats_same_as_l1'].lower()=='true':
		args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']
	else:
		args.gcn_parameters['layer_2_feats'] =random_param_value(args.gcn_parameters['layer_2_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
	args.gcn_parameters['lstm_l1_feats'] =random_param_value(args.gcn_parameters['lstm_l1_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
	if args.gcn_parameters['lstm_l2_feats_same_as_l1'] or args.gcn_parameters['lstm_l2_feats_same_as_l1'].lower()=='true':
		args.gcn_parameters['lstm_l2_feats'] = args.gcn_parameters['lstm_l1_feats']
	else:
		args.gcn_parameters['lstm_l2_feats'] =random_param_value(args.gcn_parameters['lstm_l2_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
	args.gcn_parameters['cls_feats']=random_param_value(args.gcn_parameters['cls_feats'], args.gcn_parameters['cls_feats_min'], args.gcn_parameters['cls_feats_max'], type='int')
	return args

def build_dataset(args):
	"""
	构造数据集
	"""
	if args.data == 'bitcoinotc' or args.data == 'bitcoinalpha':
		if args.data == 'bitcoinotc':
			args.bitcoin_args = args.bitcoinotc_args
		elif args.data == 'bitcoinalpha':
			args.bitcoin_args = args.bitcoinalpha_args
		return bc.bitcoin_dataset(args)
	# elif args.data == 'aml_sim':
	# 	return aml.Aml_Dataset(args)
	# elif args.data == 'elliptic':
	# 	return ell.Elliptic_Dataset(args)
	elif args.data == 'elliptic_temporal':
		return ell_temp.Elliptic_Temporal_Dataset(args)
	elif args.data == 'uc_irv_mess':
		return ucim.Uc_Irvine_Message_Dataset(args)
	# elif args.data == 'dbg':
	# 	return dbg.dbg_dataset(args)
	# elif args.data == 'colored_graph':
	# 	return cg.Colored_Graph(args)
	elif args.data == 'autonomous_syst':
		return aus.Autonomous_Systems_Dataset(args)
	elif args.data == 'reddit':
		return rdt.Reddit_Dataset(args)
	elif args.data.startswith('sbm'):
		if args.data == 'sbm20':
			args.sbm_args = args.sbm20_args
		elif args.data == 'sbm50':
			args.sbm_args = args.sbm50_args
		return sbm.sbm_dataset(args)
	else:
		raise NotImplementedError('only arxiv has been implemented')

def build_tasker(args,dataset):
	if args.task == 'link_pred':
		return lpt.Link_Pred_Tasker(args,dataset)
	elif args.task == 'edge_cls':
		return ect.Edge_Cls_Tasker(args,dataset)
	elif args.task == 'node_cls':
		return nct.Node_Cls_Tasker(args,dataset)
	elif args.task == 'static_node_cls':
		return nct.Static_Node_Cls_Tasker(args,dataset)

	else:
		raise NotImplementedError('still need to implement the other tasks')

def build_gcn(args,tasker):
	gcn_args = u.Namespace(args.gcn_parameters)
	gcn_args.feats_per_node = tasker.feats_per_node
	if args.model == 'gcn':
		return mls.Sp_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	elif args.model == 'skipgcn':
		return mls.Sp_Skip_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	elif args.model == 'skipfeatsgcn':
		return mls.Sp_Skip_NodeFeats_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	else:
		assert args.num_hist_steps > 0, 'more than one step is necessary to train LSTM'
		if args.model == 'lstmA':
			return mls.Sp_GCN_LSTM_A(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'gruA':
			return mls.Sp_GCN_GRU_A(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'lstmB':
			return mls.Sp_GCN_LSTM_B(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'gruB':
			return mls.Sp_GCN_GRU_B(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		# elif args.model == 'egcn':
		# 	return egcn.EGCN(gcn_args, activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'egcn_h':
			return egcn_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
		elif args.model == 'skipfeatsegcn_h':
			return egcn_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device, skipfeats=True)
		elif args.model == 'egcn_o':
			return egcn_o.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
		else:
			raise NotImplementedError('need to finish modifying the models')

def build_classifier(args,tasker):
	"""
	构建分类器实例。模型中分类器和嵌入模型是分开优化的，因此它是输入是嵌入模型的输出
	
	"""
	# 对于节点级任务，仅需要节点的嵌入向量，对于边级任务则需要一对节点的嵌入向量，
	# 故针对node_cls任务，mult=1,对于边级任务mult=2
	if 'node_cls' == args.task or 'static_node_cls' == args.task:
		mult = 1
	else:
		mult = 2
	# 如果是gru,lstm模型，它的输入是lstm_l2的输出
	if 'gru' in args.model or 'lstm' in args.model:
		in_feats = args.gcn_parameters['lstm_l2_feats'] * mult
	# 如果是skipgcn等，则需要将原始特征与嵌入特征合并
	elif args.model == 'skipfeatsgcn' or args.model == 'skipfeatsegcn_h':
		in_feats = (args.gcn_parameters['layer_2_feats'] + args.gcn_parameters['feats_per_node']) * mult
	# 其他情况，输入是最后一层的输出
	else:
		in_feats = args.gcn_parameters['layer_2_feats'] * mult

	return mls.Classifier(args,in_features = in_feats, out_features = tasker.num_classes).to(args.device)

if __name__ == '__main__':
	parser = u.create_parser()
	args = u.parse_args(parser)

	global rank, wsize, use_cuda # rank,world_size, use_cuda都是torch.distributed的关键字
	args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
	args.device='cpu'
	if args.use_cuda:
		args.device='cuda'
	print ("use CUDA:", args.use_cuda, "- device:", args.device)
	try:
		# torch.distributed支持GLOO，NCLL和MPI三种后端
		# backend(str或Backend)：--要使用的backend。依赖于构筑时间配置，mpi, gloo, nccl作为有效值。这个参数以小写字符串表示(例如，“gloo”)，这也能通过Backend属性(例如，Backend.GLOO)来访问。
		# 如果利用nccl在每个机器上使用多进程，每个进程必须独占访问它使用的每个GPU，因为在进程间共享GPU将会导致停滞。
		# init_method(str，可选)--URL指定如何初始化进程组。默认是“env://”，如果init_method和store都没有指定的时候。和store是互斥的。
		# world_size (python:int, optional) – Number of processes participating in the job. Required if store is specified
		# rank (python:int, optional) – Rank of the current process. Required if store is specified.
		# store(store，可选的)--所有工作可访问的键/值存储，用于交换连接/地址信息。与init_method互斥。
		# imeout(timedelta,可选的)--对流程组执行的操作的超时。默认值为30分钟。这适用于gloo后端。
		# 对于nccl，只有在环境变量NCCL_BLOCKING_WAIT被设置为1时才适用。
		# 为了使得backend == Backend.MPI，PyTorch需要在支持MPI的系统上从源代码构建。这对NCCL同样适用。
		# https://blog.csdn.net/weixin_36670529/article/details/104018195
		dist.init_process_group(backend='mpi') #, world_size=4
		# 返回当前进程组的等级。在一个分布式进程组内，rank是一个独特的识别器分配到每个进程。它们通常是从0到world_size的连续的整数。
		rank = dist.get_rank()
		# 返回当前进程组的进程数。
		wsize = dist.get_world_size()
		print('Hello from process {} (out of {})'.format(dist.get_rank(), dist.get_world_size()))
		if args.use_cuda:
			torch.cuda.set_device(rank )  # are we sure of the rank+1????
			print('using the device {}'.format(torch.cuda.current_device()))
	except:
		rank = 0
		wsize = 1
		print(('MPI backend not preset. Set process rank to {} (out of {})'.format(rank,
																				   wsize)))

	if args.seed is None and args.seed!='None':
		seed = 123+rank#int(time.time())+rank
	else:
		seed=args.seed#+rank
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	args.seed=seed
	args.rank=rank
	args.wsize=wsize

	# Assign the requested random hyper parameters
	args = build_random_hyper_params(args)

	#build the dataset
	dataset = build_dataset(args)
	#build the tasker
	tasker = build_tasker(args,dataset)
	#build the splitter
	splitter = sp.splitter(args,tasker)
	#build the models
	gcn = build_gcn(args, tasker)
	classifier = build_classifier(args,tasker)
	#build a loss
	cross_entropy = ce.Cross_Entropy(args,dataset).to(args.device)

	#trainer
	trainer = tr.Trainer(args,
						 splitter = splitter,
						 gcn = gcn,
						 classifier = classifier,
						 comp_loss = cross_entropy,
						 dataset = dataset,
						 num_classes = tasker.num_classes)

	trainer.train()
