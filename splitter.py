from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import utils as u

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
            
            random_perm=False
            indexes = tasker.data.nodes_with_label
            
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
            dev = DataLoader(dev, shuffle=False,**args.data_loading_params)
            
            test = static_data_split(tasker, self.test_idx, test = True)
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

    def __getitem__(self,idx):
        idx = self.indexes[idx]
        return self.tasker.get_sample(idx,test = self.test)
