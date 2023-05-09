from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import utils as u
"""
数据划分脚本，将数据划分为train,dev,test;
如果任务是静态的，按yaml文件中的预定义比例划分；然后调用static_data_split将其组装为一个迭代器，在构造为一个DataLoader，只有一个元素

如果任务是动态的，按时间先后顺序划分;然后调用data_split将其组装为一个迭代器，再构造为一个DataLoader，按时间步存在多个元素

其接受的输入为
args：即yaml文件的参数
tasker: 任务类，对于分类任务，可能存在两种类，静态任务tasker,动态任务tasker.

静态任务tasker主方法为:
get_sample:根据给定的id返回样本，为一个dict,包括id,adj,node_feats,label,mask=None

属性包括：
data：全部样本集
args:
num_classes：2
adj_matrix:
feats_per_nodes：
nodes_feats:
is_static:True


动态任务tasker主方法为：
主方法为:
get_sample:根据给定的id返回样本，为一个dict,包括id,历史的adj,node_feats,label,mask

属性包括：
data：全部样本集
args:
num_classes：2
feats_per_nodes
nodes_labels_times:
is_static:False
"""
class splitter():
    '''
    creates 3 splits
    train
    dev
    test
    '''
    def __init__(self,args,tasker):
        
        
        if tasker.is_static: #### For static datsets
            assert args.train_proportion + args.dev_proportion < 1, \
                'there\'s no space for test samples'
            #only the training one requires special handling on start, the others are fine with the split IDX.
            #! 对于静态模型，batch_size设为100000,对于动态数据设为1
            args.data_loading_params['batch_size'] = 100000
            random_perm=False
            indexes = tasker.data.nodes_labels_times[:,0] #只取indexes,即在全部节点列表中的序号
            
            if random_perm:
                perm_idx = torch.randperm(indexes.size(0))
                perm_idx = indexes[perm_idx]
            else:
                print ('tasker.data.nodes',indexes.size())
                perm_idx, _ = indexes.sort()
            #print ('perm_idx',perm_idx[:10])
            
            self.train_idx = perm_idx[:int(args.train_proportion*perm_idx.size(0))]
            self.dev_idx = perm_idx[int(args.train_proportion*perm_idx.size(0)): int((args.train_proportion+args.dev_proportion)*perm_idx.size(0))]
            self.test_idx = perm_idx[int((args.train_proportion+args.dev_proportion)*perm_idx.size(0)):]
            # print ('train,dev,test',self.train_idx.size(), self.dev_idx.size(), self.test_idx.size())
            
            train = static_data_split(tasker, self.train_idx, test = False)
           
            train = DataLoader(train, shuffle=True,**args.data_loading_params)
            
            dev = static_data_split(tasker, self.dev_idx, test = True)
            # args.data_loading_params['batch_size'] = dev.__len__()
            dev = DataLoader(dev, shuffle=False,**args.data_loading_params)
            
            test = static_data_split(tasker, self.test_idx, test = True)
            # args.data_loading_params['batch_size'] = test.__len__()
            test = DataLoader(test, shuffle=False,**args.data_loading_params)
                        
            self.tasker = tasker
            self.train = train
            self.dev = dev
            self.test = test
            
            
        else: #### For datsets with time
            assert args.train_proportion + args.dev_proportion < 1, \
                'there\'s no space for test samples'
            #only the training one requires special handling on start, the others are fine with the split IDX.
            start = tasker.data.min_time + args.num_hist_steps #-1 + args.adj_mat_time_window
            end = args.train_proportion
            
            end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
            train = data_split(tasker, start, end, test = False)
            train = DataLoader(train,**args.data_loading_params)
    
            start = end
            end = args.dev_proportion + args.train_proportion
            end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
            if args.task == 'link_pred':
                # 返回的是一个生成器
                dev = data_split(tasker, start, end, test = True, all_edges=True)
            else:
                dev = data_split(tasker, start, end, test = True)

            dev = DataLoader(dev,num_workers=args.data_loading_params['num_workers'])
            
            start = end
            
            #the +1 is because I assume that max_time exists in the dataset
            end = int(tasker.max_time) + 1
            if args.task == 'link_pred':
                test = data_split(tasker, start, end, test = True, all_edges=True)
            else:
                test = data_split(tasker, start, end, test = True)
                
            test = DataLoader(test,num_workers=args.data_loading_params['num_workers'])
            
            print ('Dataset splits sizes:  train',len(train), 'dev',len(dev), 'test',len(test))
            
            
            
            self.tasker = tasker
            self.train = train
            self.dev = dev
            self.test = test
        


class data_split(Dataset):
    def __init__(self, tasker, start, end, test, **kwargs):
        '''
        start and end are indices indicating what items belong to this split
        tasker: 任务实例,node_cls_tasker,link_pred_tasker,edge_clas_tasker
        start:开始的时间点
        end: 结束的时间点
        test: 对于link_pred_tasker,需定义test参数来控制训练和测试中负采样的数量
        因此训练是逐步自回归的，数据每次增加一个时间步，每次都加载历史数据
        '''
        self.tasker = tasker
        self.start = start
        self.end = end
        self.test = test
        self.kwargs = kwargs

    def __len__(self):
        return self.end-self.start

    def __getitem__(self,idx):
        #* 由此可以看出训练是逐步自回归的，数据每次增加一个时间步，每次都加载以前的全部历史数据
        #* 其返回的是一个生成器
        idx = self.start + idx
        t = self.tasker.get_sample(idx, test = self.test, **self.kwargs)
        return t


class static_data_split(Dataset):
    def __init__(self, tasker, indexes, test):
        '''
        start and end are indices indicating what items belong to this split
        tasker: 任务实例,node_cls_tasker,link_pred_tasker,edge_clas_tasker
        indexes: 时间点的列表
        test: 对于link_pred_tasker,需定义test参数来控制训练和测试中负采样的数量
        '''
        self.tasker = tasker
        self.indexes = indexes
        self.test = test
        self.adj_matrix = tasker.adj_matrix

    def __len__(self):
        return len(self.indexes)
    # 迭代器
    def __getitem__(self,idx):
        # print(idx) # 把所有数据作为一个batch
        # idx_raw = self.indexes[idx]
        return self.tasker.get_sample(idx,test = self.test)
