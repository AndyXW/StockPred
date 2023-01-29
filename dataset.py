import numpy as np

import torch
import torch.nn as nn

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, stock_data, stock_label) -> None:
        super().__init__()
        self.data = stock_data
        self.label = stock_label
        self.label_dim = len(self.label.shape)
    
    def __getitem__(self, index):
        if self.label_dim == 1:
            return self.data[index, :, :].to(torch.float32), self.label[index].to(torch.float32)
        else:
            return self.data[index, :, :], self.label[index, :]

    
    def __len__(self):
        return self.data.shape[0]


# class StockDataset(torch.utils.data.Dataset):
#     def __init__(self, stock_list, history_length=20, stride=10) -> None:
#         super().__init__()
#         self.stock_list = stock_list
#         self.history_length = history_length
#         self.stride = stride
#         self.length = len(stock_list[0].dates)
#         self.max_inner_idx = (self.length - self.history_length) // self.stride
    
#     def __getitem__(self, index):
#         stock_id =  index // self.max_inner_idx#(index * self.stride + self.history_length) // self.length
#         inner_stock_idx = index - stock_id * self.max_inner_idx
#         # print(stock_id, inner_stock_idx)
#         try:
#             feature = self.stock_list[stock_id].data[inner_stock_idx*self.stride: inner_stock_idx*self.stride+self.history_length]
#         except:
#             print(stock_id)
#         label = self.stock_list[stock_id].labels[inner_stock_idx*self.stride+self.history_length-1]

#         return feature, label

#     def __len__(self):
#         return (self.max_inner_idx) * len(self.stock_list)


# def my_collate(batch):
#     # print(len(batch))
#     batch_feat = []
#     batch_label = []
#     for item in batch:
#         # print(item)
#         # print(True in np.isnan(item[0]))
#         # print((True == np.isnan(np.array(item[1]))))
#         if (True in np.isnan(item[0])) or (True == np.isnan(np.array(item[1]))):
#             pass
#         else:
#             batch_feat.append(torch.from_numpy(item[0]).unsqueeze(0))
#             batch_label.append(torch.tensor(item[1]))
#     if len(batch_feat) > 0:
#         feats = torch.cat(batch_feat, dim=0)
#         labels = torch.tensor(batch_label)
#         return feats, labels
#     return None, None