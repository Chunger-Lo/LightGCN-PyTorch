"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Loader(BasicDataset):
    def __init__(self,config = world.config,path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        print(config['test_date'])

        if world.dataset == 'sc':
            # dir_name = '/fake'
            dir_name = config['test_date']
            user_item_train_file = path + '/date='+ config['test_date']  +'/train_user_item_retain.txt'
            user_item_test_file = path + '/date='+ config['test_date']  + '/test_user_item_retain.txt'
            user_subtag_train_file = path + '/date='+ config['test_date']  +'/user_subtag.txt'
            user_subtag_test_file = path + '/date='+ config['test_date']  + '/user_subtag.txt'
            item_subtag_train_file = path +'/date='+ config['test_date']  +'/item_subtag.txt'
            item_subtag_test_file = path + '/date='+ config['test_date']  + '/item_subtag.txt'
        else:
            print('wrong dataset')
        self.path = path
        # trainUniqueUsers, trainItem, trainUser, trainSubtag = [], [], [], []
        # testUniqueUsers, testItem, testUser, testSubtag = [], [], [], []
        trainUniqueUsers, testUniqueUsers = [], []
        # self.traindataSize = 0
        # self.testDataSize = 0
        self.n_user = 0
        self.m_item = 0
        self.r_subtag = 0
        #
        self.traindataSize_user_item = 0
        self.testdataSize_user_item = 0
        self.traindataSize_user_subtag = 0
        self.testdataSize_user_subtag = 0
        self.traindataSize_item_subtag = 0
        self.testdataSize_item_subtag = 0

        # user-item
        print(f'Loading {user_item_train_file}')
        print(f'Loading {user_item_test_file}')
        trainUser = []
        trainItem = []
        n_interactions_train = 0 #number of interactions
        nuser_train_user_item = 0
        with open(user_item_train_file) as f:
            for l in f.readlines():
                nuser_train_user_item += 1
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    ## create connected pairs
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    if len(items)>0:
                        self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    n_interactions_train += len(items)
                # if world.mode == 'fastdebug':
                #     if nuser_train == 10000:
                #         break
        # trainUser_user_item, trainUser_user_subtag = []
        # trainItem_item_subtag, trainItem_user_item = []
        # trainSubtag_user_subtag, trainSubtag_item_subtag = []
        # testUser_user_item, testUser_user_subtag = []
        # testItem_item_subtag, testItem_user_item = []
        # testSubtag_user_subtag, testSubtag_item_subtag = []
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser_user_item = np.array(trainUser)
        self.trainItem_user_item = np.array(trainItem)
        self.traindataSize_user_item = n_interactions_train
        self.__testDict = {}
        testUser = []
        testItem = []      
        nuser_test_user_item = 0 
        n_interactions_test = 0 #number of interactions
        with open(user_item_test_file) as f:
            for l in f.readlines():
                nuser_test_user_item += 1
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.__testDict[uid] = items ## add
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    if len(items)>0:
                        self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    n_interactions_test += len(items)
                # if world.mode == 'fastdebug':
                #     if nuser_test == 1024:
                #         break
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser_user_item = np.array(testUser)
        self.testItem_user_item = np.array(testItem)
        self.testdataSize_user_item = n_interactions_test
         
        # user-subtag
        trainUser = []
        trainSubtag = []
        nuser_train_user_subtag = 0
        n_interactions_train = 0 #number of interactions
        with open(user_subtag_train_file) as f:
            for l in f.readlines():
                nuser_train_user_subtag += 1
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    subtags = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(subtags))
                    trainSubtag.extend(subtags)
                    if len(subtags)>0:
                        self.r_subtag = max(self.r_subtag, max(subtags))
                    # self.n_user = max(self.n_user, uid)
                        n_interactions_train += len(subtags)
                # if world.mode == 'fastdebug':
                #     if nuser_train_user_subtag == 10000:
                #         break
        self.trainUser_user_subtag = np.array(trainUser)
        self.trainSubtag_user_subtag = np.array(trainSubtag)
        self.traindataSize_user_subtag = n_interactions_train

        testUser = []
        testSubtag = []  
        n_interactions_test = 0      
        with open(user_subtag_test_file) as f:
            for l in f.readlines():
                # nuser_test += 1
                # self.n_user += 1
                # self.n_train += 1
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    subtags = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    testUser.extend([uid] * len(subtags))
                    testSubtag.extend(subtags)
                    if len(subtags)>0:
                        self.r_subtag = max(self.r_subtag, max(subtags))
                        n_interactions_test += len(subtags)
                # if world.mode == 'fastdebug':
                #     if nuser_test == 10000:
                #         break
        self.testUser_user_subtag = np.array(testUser)
        self.testSubtag_user_subtag = np.array(testSubtag)
        self.testdataSize_user_subtag = n_interactions_test
        # item-subtag
        trainItem = []
        trainSubtag = []
        n_interactions_train = 0
        with open(item_subtag_train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    subtags = [int(i) for i in l[1:]]
                    item_id = int(l[0])
                    trainItem.extend([item_id] * len(subtags))
                    trainSubtag.extend(subtags)
                    if len(subtags)>0:
                        self.r_subtag = max(self.r_subtag, max(subtags))
                    # self.n_user = max(self.n_user, uid)
                        n_interactions_train += len(subtags)
        self.trainItem_item_subtag = np.array(trainItem)
        self.trainSubtag_item_subtag = np.array(trainSubtag)
        self.traindataSize_item_subtag = n_interactions_train
        ### Loaded and print results
        self.n_user += 1
        self.m_item += 1
        self.r_subtag += 1
        
        print(f'Number of user (user-subtag):{nuser_train_user_item}')
        print(f'Number of user (user-subtag): {nuser_train_user_subtag}')
        nuser_train = nuser_train_user_item
        nuser_test = nuser_test_user_item

        print(f"{self.traindataSize_user_item} interactions of user-item for training")
        print(f"{self.testdataSize_user_item} interactions of user-item for testing")
        print(f"{self.traindataSize_user_subtag} interactions of user-subtag for training")
        print(f"{self.testdataSize_user_subtag} interactions of user-subtag for testing")
        print(f"{self.traindataSize_item_subtag} interactions of item-subtag for training")
        # print(f"{self.testdataSize_user_item} interactions of item-subtag for testing")
        print(f"{nuser_train} number of users in train")
        print(f"{nuser_test} number of users in test")
        print(f"{self.n_user} number of users")
        print(f"{self.m_item} number of items")
        print(f"{self.r_subtag} number of subtags")
        # print(f"{world.dataset} Sparsity : {(self.traindataSize_user_item + self.testdataSize_user_item) / self.n_users / self.m_items}")
        ### Create grpah
        self.Graph = None
        # (users,items), bipartite graph 
        print(f'data: {np.ones(len(self.trainUser_user_item))}')
        print(f'index row: {self.trainUser_user_item}') #max
        print(f'index row max: {max(self.trainUser_user_item)}') #max
        print(f'index col: {self.trainItem_user_item}')
        print(f'shape of row: {self.n_user}') # shape 0 
        print(f'shape of col: {self.m_item}')
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser_user_item)), 
                                        (self.trainUser_user_item, self.trainItem_user_item)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # (users,subtags), bipartite graph
        # self.trainUser_user_subtag = [x - 1 for x in self.trainUser_user_subtag]
        print(f'data: {np.ones(len(self.trainUser_user_subtag))}')
        print(f'index row: {self.trainUser_user_subtag}')
        print(f'index row max: {max(self.trainUser_user_subtag)}') #max
        print(f'index col: {self.trainSubtag_user_subtag}')
        print(f'shape of row: {self.n_user}') # shape 0 
        print(f'shape of col: {self.r_subtag}')
        self.UserSubtagNet = csr_matrix((np.ones(len(self.trainUser_user_subtag)), 
                                        (self.trainUser_user_subtag, self.trainSubtag_user_subtag)),
                                      shape=(self.n_user, self.r_subtag))
        # (items, subtags), bipartite graph
        print(f'data: {np.ones(len(self.trainItem_item_subtag))}')
        print(f'index row: {self.trainItem_item_subtag}') #max
        print(f'index row max: {max(self.trainItem_item_subtag)}') #max
        print(f'index col: {self.trainSubtag_item_subtag}')
        # print(f'row vec length: {len(self.trainItem_item_subtag)}')
        # print(f'col vec length: {len(self.trainSubtag_item_subtag)}')
        print(f'shape of row: {self.m_item}') # shape 0 
        print(f'shape of col: {self.r_subtag}')
        self.ItemSubtagNet = csr_matrix((np.ones(len(self.trainItem_item_subtag)), 
                                        (self.trainItem_item_subtag, self.trainSubtag_item_subtag)),
                                      shape=(self.m_item, self.r_subtag))
        ## examine
        print(f"{self.n_user} number of users")
        print(f"{self.m_item} number of items")
        print(f"{self.r_subtag} number of subtags")
        # pre-calculate
        print('getting UserPosItems')
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        # print('getting ItemPosSubtag')
        # self._allPos_subtag_byItem = self.getItemPosSubtags(list(range(self.m_item)))
        # print('getting UserPosSubtag')
        # self._allPos_subtag_byUser = self.getUserPosSubtags(list(range(self.n_user)))
        # self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    @property
    def trainDataSize(self):
        # return self.traindataSize
        return self.traindataSize_user_item
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos
    @property
    def allPos_subtag(self):
        return self._allPos_subtag
    @property
    def allPos_subtag(self):
        return self._allPos_subtag

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                # end = self.n_users + self.m_items
                end = self.n_users + self.m_items + self.r_subtag
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self, config = world.config):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                if world.dataset == 'sc': 
                    re_adj_mat = sp.load_npz(self.path +'/date='+ config['test_date']+'/s_pre_adj_mat.npz')
                    print("successfully loaded...")
                    norm_adj = pre_adj_mat
                else:
                    pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                    print("successfully loaded...")
                    norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                # adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = sp.dok_matrix((self.n_users + self.m_items + self.r_subtag, self.n_users + self.m_items + self.r_subtag), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R_user_item = self.UserItemNet.tolil()
                R_user_subtag = self.UserSubtagNet.tolil()
                R_item_subtag = self.ItemSubtagNet.tolil()
                print(f'shape of lil: {R_user_item.shape}')
                print(f'shape of lil: {R_item_subtag.shape}')
                adj_mat[:self.n_users, self.n_users:self.n_users+self.m_items] = R_user_item
                adj_mat[self.n_users:self.n_users+self.m_items, :self.n_users] = R_user_item.T
                # print(f'shape of target: {adj_mat[self.n_users:self.n_users+self.m_items, self.n_users+self.m_items:].shape}')
                adj_mat[self.n_users:self.n_users+self.m_items, self.n_users+self.m_items:] = R_item_subtag
                adj_mat[self.n_users+self.m_items:, self.n_users:self.n_users+self.m_items] = R_item_subtag.T
                adj_mat[:self.n_users, self.n_users+self.m_items:] = R_user_subtag
                adj_mat[self.n_users+self.m_items:, :self.n_users] = R_user_subtag.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten() 
                # note: what if rowsum have value = 0
                # set inf to 0
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                ## tilde A
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")

                if world.dataset == 'sc':
                    sp.save_npz(self.path + '/date='+ config['test_date']+'/s_pre_adj_mat.npz', norm_adj)
                else:
                    sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        # for i, item in enumerate(self.testItem):
        for i, item in enumerate(self.testItem_user_item):
            print(f'building {i}th user with item: {item}')
            user = self.testUser_user_item[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        print(len(list(test_data.keys())))

        # test_data = {}
        # for i, user in enumerate(self.testUser_user_item):
        #     item =  self.testItem_user_item[i]
        #     if test_data.get(user):
        #         test_data[user].append(item)
        #     else:
        #         test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    # def getUserPosSubtags(self, users):
    #     posSubtags = []
    #     for user in users:
    #         posSubtags.append(self.UserSubtagNet[user].nonzero()[1])
    #     return posSubtags
    # def getItemPosSubtags(self, users):
    #     posSubtags = []
    #     for item in items:
    #         posSubtags.append(self.ItemSubtagNet[item].nonzero()[1])
    #     return posSubtags

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
