import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


# 时间序列数据集构造类
class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config=None):
        self.raw_data = raw_data  # 原始数据
        self.config = config      # 配置参数
        self.edge_index = edge_index  # 图结构的边索引（用于图神经网络）
        self.mode = mode          # 模式：train或test

        # 分离特征和标签
        # 假设raw_data的最后一行是标签，其余行是特征
        x_data = raw_data[:-1]    # 特征数据
        labels = raw_data[-1]     # 标签数据

        data = x_data

        # 转换为PyTorch张量
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        # 处理数据，生成训练样本
        self.x, self.y, self.labels = self.process(data, labels)

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.x)

    def process(self, data, labels):
        """
        处理原始数据，生成滑动窗口样本
        Args:
            data: 特征数据张量
            labels: 标签数据张量
        Returns:
            x: 输入特征 [样本数, 节点数, 窗口大小]
            y: 预测目标 [样本数, 节点数]
            labels: 样本标签 [样本数]
        """
        x_arr, y_arr = [], []     # 存储输入特征和目标值的列表
        labels_arr = []           # 存储标签的列表

        # 从配置中获取滑动窗口参数
        slide_win, slide_stride = [self.config[k] for k
                                   in ['slide_win', 'slide_stride']]
        is_train = self.mode == 'train'  # 判断是否为训练模式

        node_num, total_time_len = data.shape  # 获取节点数和总时间长度 data.shape = torch.Size([27, 1565])

        # 根据模式确定滑动范围
        # 训练模式：使用步长滑动，减少样本重叠
        # 逐点滑动，最大化利用数据
        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)

        for i in rang:
            # 提取滑动窗口内的历史数据作为输入特征 [节点数, 窗口大小]  [i-slide_win, i) 时间范围内的数据
            ft = data[:, i-slide_win:i]
            # 提取窗口后的当前时刻数据作为预测目标 [节点数] [i]时刻数据
            tar = data[:, i]

            x_arr.append(ft)      # 添加到输入特征列表
            y_arr.append(tar)     # 添加到目标值列表

            labels_arr.append(labels[i])  # 添加当前时间点的标签

        # 将列表堆叠成张量
        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()
        labels = torch.Tensor(labels_arr).contiguous()

        return x, y, labels

    def __getitem__(self, idx):
        """获取指定索引的数据样本"""
        feature = self.x[idx].float()    # 输入特征 [节点数, 窗口大小]
        y = self.y[idx].float()          # 预测目标 [节点数]
        edge_index = self.edge_index.long()  # 图结构的边索引
        label = self.labels[idx].float()    # 样本标签

        return feature, y, label, edge_index