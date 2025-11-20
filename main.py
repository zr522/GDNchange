# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from models.MSTCN import TCN1d
from models.search import  search_with_genetic_algorithm
from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset


from models.GDN import GDN

from train import train
from test  import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random
import logging
log_dir = './log'
os.makedirs(log_dir, exist_ok=True)


class Main():
    def __init__(self, train_config, env_config, debug=False):

        logging.basicConfig(
        filename=os.path.join(log_dir, 'training_log.txt'),  # 日志文件名
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
        )

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None
        self.MSConv = None

        dataset = self.env_config['dataset'] 
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)
       
        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.edge_index_sets = [fc_edge_index]  # 为模型初始化时设置
        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())


        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)


        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0)


        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        # 引入时序卷积
        if train_config['use_tcn']:
            self.MSConv = TCN1d(
                feature_num=len(feature_map),
                # kernel_size=5,
                # dilation=2
            )

        self.model = GDN(edge_index_sets=self.edge_index_sets,
                node_num=len(feature_map),
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk'],
                MSConv=self.MSConv,
                use_nas=train_config['use_nas'], # <- 开启NAS
                search_space={  # 设置搜索空间
                    "depth": [1, 2, 3, 4, 5],
                    "op_candidates": ["GraphLayer", "GCN", "GAT", "SAGE", "GraphSAGE"],
                    "hidden_candidates": [32, 64, 128, 256],
                    "skip_candidates": ["none", "residual"],
                }
                # search_space = {
                #     "depth": [1],
                #     "op_candidates": ["GraphLayer"],       # 只允许 GraphLayer
                #     "hidden_candidates": [64],      # 只允许原始 dim
                #     "skip_candidates": ["none"]           # 不加残差
                # }
            ).to(self.device)



    def run(self):
        # 只有在启用NAS时才进行架构搜索
        if self.train_config.get('use_nas', False):
            # 先做随机搜索（例如 20个候选，每个训练10个epoch）
            def make_model():
                # 构建新的 GDN（保持与上面一致），用于评测单个候选
                return GDN(
                    self.edge_index_sets, len(self.feature_map),
                    dim=self.train_config['dim'],
                    input_dim=self.train_config['slide_win'],
                    out_layer_num=self.train_config['out_layer_num'],
                    out_layer_inter_dim=self.train_config['out_layer_inter_dim'],
                    topk=self.train_config['topk'],
                    use_nas=self.train_config['use_nas'],  # 启用NAS
                    search_space={  # 设置搜索空间
                        "depth": [1,2, 3, 4 , 5],
                        "op_candidates": ["GraphLayer", "GCN", "GAT", "SAGE", "GraphSAGE"],
                        "hidden_candidates": [32, 64, 128, 256],
                        "skip_candidates": ["none", "residual"],
                    }
                    # search_space = {
                    #     "depth": [1],
                    #     "op_candidates": ["GraphLayer"],       # 只允许 GraphLayer
                    #     "hidden_candidates": [64],      # 只允许原始 dim
                    #     "skip_candidates": ["none"]           # 不加残差
                    # }
                ).to(self.device)

            # best_arch = random_search(
            #     make_model_fn=make_model,
            #     train_dataloader=self.train_dataloader,
            #     val_dataloader=self.val_dataloader,
            #     base_train_config=self.train_config,
            #     num_samples=20,
            #     short_epochs=15,
            #     device=str(self.device)
            # )
            # best_arch = search_with_genetic_algorithm(make_model_fn=make_model, train_dataloader=self.train_dataloader,
            #                                           val_dataloader=self.val_dataloader,
            #                                           base_train_config=self.train_config, num_generations=2,
            #                                           population_size=20, mutation_rate=0.3, device="cuda",
            #                                           tmp_save_dir="./pretrained/nas_tmp/")
            best_arch = {'depth': 4,
                    'ops': ['GraphLayer', 'GAT', 'GraphSAGE', 'SAGE'],
                    'hidden_dims': [256, 128, 256, 128],
                    'skip': 'none'}
            print(">>> Best arch from NAS:", best_arch)


            # 只有当best_arch不为None且模型支持NAS时才调用build_arch
            if best_arch is not None:
                # 优先使用多分支 NAS（searchable_gnn_layers）
                if getattr(self.model, "searchable_gnn_layers", None) is not None:
                    print("Apply best_arch to all NAS branches (searchable_gnn_layers)")
                    for i, layer in enumerate(self.model.searchable_gnn_layers):
                        print(f"  -> build_arch for NAS layer {i}")
                        layer.build_arch(best_arch)

                # 兼容单分支 NAS（老版本）
                elif getattr(self.model, "searchable_gnn", None) is not None:
                    print("Apply best_arch to single NAS layer (searchable_gnn)")
                    self.model.searchable_gnn.build_arch(best_arch)

                else:
                    print("WARNING: Model has no searchable_gnn_layers/searchable_gnn, skip build_arch")
        else:
            print("NAS is disabled. Skipping architecture search.")
            best_arch = None

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]

            self.train_log = train(self.model, model_save_path, 
                config = train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset']
            )
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)

        self.get_score(self.test_result, self.val_result)

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()
    
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1) 
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)


        print('=========================** Result **============================\n')
        logging.info('=========================** Result **============================')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')

        logging.info(f'F1 score: {info[0]}')
        logging.info(f'precision: {info[1]}')
        logging.info(f'recall: {info[2]}')


    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%Y-%m-%d_%H-%M-%S')
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='swat')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=20)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'use_tcn': True,
        'use_nas': False
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }
    

    main = Main(train_config, env_config, debug=False)
    main.run()





