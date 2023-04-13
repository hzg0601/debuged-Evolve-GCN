import torch
import utils as u
import logger
import time
import pandas as pd
import numpy as np
import os


log_file = './embs/'
if not os.path.exists(log_file):
    os.mkdir(log_file)
   
    
class Trainer():
	def __init__(self,args, splitter, gcn, classifier, comp_loss, dataset, num_classes):
		self.args = args
		self.splitter = splitter # 数据集的splitter实例
		self.tasker = splitter.tasker #tasker也是splitter的传入参数
		self.gcn = gcn # 图神经网络模型实例
		self.classifier = classifier # 分类器的实例
		self.comp_loss = comp_loss # 损失计算方法

		self.num_nodes = dataset.num_nodes
		self.data = dataset
		self.num_classes = num_classes

		self.logger = logger.Logger(args, self.num_classes)

		self.init_optimizers(args)

		if self.tasker.is_static:
			adj_matrix = u.sparse_prepare_tensor(self.tasker.adj_matrix, torch_size = [self.num_nodes], ignore_batch_dim = False)
			self.hist_adj_list = [adj_matrix]
			self.hist_ndFeats_list = [self.tasker.nodes_feats.float()]

	def init_optimizers(self,args):
		"""
		定义gcn和分类器的优化器，gcn和分类器是分开优化的
		"""
		params = self.gcn.parameters()
		self.gcn_opt = torch.optim.Adam(params, lr = args.learning_rate)
		params = self.classifier.parameters()
		self.classifier_opt = torch.optim.Adam(params, lr = args.learning_rate)
		self.gcn_opt.zero_grad()
		self.classifier_opt.zero_grad()

	def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
		torch.save(state, filename)

	def load_checkpoint(self, filename, model):
		if os.path.isfile(filename):
			print("=> loading checkpoint '{}'".format(filename))
			checkpoint = torch.load(filename)
			epoch = checkpoint['epoch']
			self.gcn.load_state_dict(checkpoint['gcn_dict'])
			self.classifier.load_state_dict(checkpoint['classifier_dict'])
			self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
			self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
			self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
			return epoch
		else:
			self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
			return 0

	def train(self):
		"""
		多轮训练方法
		"""
		self.tr_step = 0
		best_eval_valid = 0
		eval_valid = 0
		epochs_without_impr = 0

		for e in range(self.args.num_epochs):
			eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, 'TRAIN', grad = True)
			# 每次训练后都进行验证
			if len(self.splitter.dev)>0 and e>self.args.eval_after_epochs:
				eval_valid, _ = self.run_epoch(self.splitter.dev, e, 'VALID', grad = False)
				if eval_valid>best_eval_valid:
					best_eval_valid = eval_valid
					epochs_without_impr = 0 # 如果表现有改善则定义epochs_without_impr为零
					print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Best valid measure:'+str(eval_valid))
				else:
					epochs_without_impr+=1 
					# 如果表现无改善则定义epochs_without_impr+1，epochs_without_impr超过最大
					# early_stop的阈值，则停止训练
					if epochs_without_impr>self.args.early_stop_patience:
						print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Early stop.')
						break
			#? 每次训练后既验证也测试
			if len(self.splitter.test)>0 and eval_valid==best_eval_valid and e>self.args.eval_after_epochs:
				eval_test, _ = self.run_epoch(self.splitter.test, e, 'TEST', grad = False)
				print(f'the test performance of current epoch --{e}-- is:{eval_test}')

				if self.args.save_node_embeddings and self.tasker.is_static:
					self.save_node_embs_csv(nodes_embs, self.splitter.train_idx, log_file+'_train_nodeembs.csv.gz')
					self.save_node_embs_csv(nodes_embs, self.splitter.dev_idx, log_file+'_valid_nodeembs.csv.gz')
					self.save_node_embs_csv(nodes_embs, self.splitter.test_idx, log_file+'_test_nodeembs.csv.gz')
				elif self.args.save_node_embeddings and not self.tasker.is_static:
					node_embs_numpy = nodes_embs.cpu().detach().numpy()
					pd.DataFrame(node_embs_numpy).to_csv(log_file+f"{self.args.model}_{self.args.data}.csv.gz", header=None, index=None, compression='gzip')
		print(f"the the performance of last training is: {eval_test}")
	def run_epoch(self, split, epoch, set_name, grad):
		"""
		单轮训练方法
		split->torch.DataLoader: 数据集，
		epoch: 训练的轮次
		set_name: 数据集名称,TRAIN,VALID,TEST
		grad: 是否进行梯度追踪，对于训练值为True,验证和测试为False
		"""
		t0 = time.time()
		log_interval=999
		if set_name=='TEST':
			log_interval=1
		self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)
		torch.cuda.empty_cache()
		torch.set_grad_enabled(grad)
		#* 训练是逐步回归的，每次增加一个时间步，加载该时间点之前的全部历史数据
		for s in split:
			if self.tasker.is_static:
				s = self.prepare_static_sample(s)
			else:
				s = self.prepare_sample(s)

			predictions, nodes_embs = self.predict(s.hist_adj_list,
												   s.hist_ndFeats_list, #? 时变属性
												   s.label_sp['idx'],
												   s.node_mask_list)

			loss = self.comp_loss(predictions,s.label_sp['vals'])
			# print(loss)
			if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
			else:
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())
			if grad:
				self.optim_step(loss)

		torch.set_grad_enabled(True)
		eval_measure = self.logger.log_epoch_done()

		return eval_measure, nodes_embs

	def predict(self,hist_adj_list,hist_ndFeats_list,node_indices,mask_list):
		"""
		调用gcn模型训练给定时间点的历史数据

		"""
		# 获取所有节点的嵌入向量
		# 每一层都进行逐步回归，保留历史的节点嵌入列表，用作下一层的输入，
		# 以最后一层输出的最后一个时间点作为最终的输出
		nodes_embs = self.gcn(hist_adj_list,
							  hist_ndFeats_list,
							  mask_list)

		predict_batch_size = 100000
		gather_predictions=[]
		# 预测是分批进行的，
		# 毫无意义
		for i in range(1 +(node_indices.size(1)//predict_batch_size)):
			cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
			predictions = self.classifier(cls_input)
			gather_predictions.append(predictions)
		gather_predictions=torch.cat(gather_predictions, dim=0)
		return gather_predictions, nodes_embs

	def gather_node_embs(self,nodes_embs,node_indices):
		cls_input = []

		for node_set in node_indices:
			cls_input.append(nodes_embs[node_set])
		return torch.cat(cls_input,dim = 1)

	def optim_step(self,loss):
		"""
		gcn模型和分类器是分开优化的
		"""
		self.tr_step += 1
		loss.backward()

		if self.tr_step % self.args.steps_accum_gradients == 0:
			self.gcn_opt.step()
			self.classifier_opt.step()

			self.gcn_opt.zero_grad()
			self.classifier_opt.zero_grad()


	def prepare_sample(self,sample):
		"""
		将数据转换为torch.tensor，并放到device上
		sample:包含了给定时间点前，时间窗口内，所有数据的列表的dict
  		"""
		sample = u.Namespace(sample)
		for i,adj in enumerate(sample.hist_adj_list):
			adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])
			sample.hist_adj_list[i] = adj.to(self.args.device)
			# 调用prepare_node_feats将节点特征转换为torch.tensor
			nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

			sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
			node_mask = sample.node_mask_list[i]
			sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer

		label_sp = self.ignore_batch_dim(sample.label_sp)

		if self.args.task in ["link_pred", "edge_cls"]:
			label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
		else:
			label_sp['idx'] = label_sp['idx'].to(self.args.device)

		label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
		sample.label_sp = label_sp

		return sample

	def prepare_static_sample(self,sample):
		"""
		对静态数据集进行数据组装
		"""
		sample = u.Namespace(sample)

		sample.hist_adj_list = self.hist_adj_list

		sample.hist_ndFeats_list = self.hist_ndFeats_list

		label_sp = {}
		label_sp['idx'] =  [sample.idx]
		label_sp['vals'] = sample.label
		sample.label_sp = label_sp

		return sample

	def ignore_batch_dim(self,adj):
		if self.args.task in ["link_pred", "edge_cls"]:
			adj['idx'] = adj['idx'][0]
		adj['vals'] = adj['vals'][0]
		return adj

	def save_node_embs_csv(self, nodes_embs, indexes, file_name):
		csv_node_embs = []
		for node_id in indexes:
			orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

			csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())

		pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
		#print ('Node embs saved in',file_name)
