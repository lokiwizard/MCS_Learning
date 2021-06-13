import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader,ConcatDataset
import torch
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from collections import defaultdict

def dataRead(category, filePath, **kwargs):
    """
    :param category:  数据集类型
    :param filePath:  数据文件地址
    :return:List
    """
    if(category == "Geolife"):
        return Geolife(filePath)
    elif(category == "Taxi"):
        return Taxi(filePath, kwargs.get('mode'))

def Geolife(filePath):
    data = pd.read_table(filePath, skiprows=5)
    data = data["0"].str.split(",", expand=True)  # expand=True  可以把用分割的内容直接分列
    data = data.iloc[:, [0, 1]]
    data.columns = [[" Latitude", " Longitude"]]
    data = data.astype(float)
    locations = data.values.tolist()
    return np.array(locations)


def Taxi(filePath, mode):
    """
    :param filePath: String
    :param mode: Bool
    :return:
    """
    data = pd.read_table(filePath, header=None)
    data = data[0].str.split(",", expand=True)  # expand=True  可以把用分割的内容直接分列
    data_time = data[1].str.split(" ", expand=True)
    data_trajectory = data.loc[:, 2:].astype(float)
    data = pd.concat([data_time, data_trajectory], axis=1)
    if(mode):
        # 按日期分割
        mode = "time_split"
        timeStamp = sorted(list(set(data[0].values)))
        locationsDict = {}
        for i in timeStamp:
            locationsDict[i] = data[data[0]==i].loc[:,[3,2]].to_numpy()
        locations = locationsDict
        return locations
    else:
        # 直接输出
        mode = "direct_out"
        data = data.iloc[:, [3, 2]]
        data.columns = [[" Latitude", " Longitude"]]
        #data = data.astype(float)
        locations = data.values.tolist()
        return np.array(locations)

def minMaxNormalize(data, set_range):

    normalize = np.arange(2 * data.shape[1], dtype='float64')
    normalize = normalize.reshape(data.shape[1], 2)
    for i in range(0, data.shape[1]):
        if set_range == True:
            list = data[:, i]
            listlow, listhigh = np.percentile(list, [0, 100])
        else:
            if i == 0:
                listlow = -90
                listhigh = 90
            else:
                listlow = -180
                listhigh = 180
        normalize[i, 0] = listlow
        normalize[i, 1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta
    return (data, normalize)


def minMaxFnormalize(data, normalize):
    data = np.array(data, dtype='float64')
    # 列
    for i in range(0, data.shape[1]):
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        # 行
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = data[j, i] * delta + listlow
    return data


class GenerateTrainData(Dataset):

    def __init__(self, data, windowSize):
        super(GenerateTrainData, self).__init__()
        self.windowSize = windowSize
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.windowSize
    
    def __getitem__(self, i):
        features = torch.tensor(self.data[i:i + self.windowSize, :], dtype=torch.float32,
                                device=device)
        labels = torch.tensor(self.data[i + self.windowSize, :], dtype=torch.float32,
                              device=device)
        return features, labels
    
"""
以下两个函数适用于成都的数据集
使用案例：
data = Getdata("D:\dataset\gps_01-10\gps_20161101")
data的形式是一个字典，key为taxi_id, value为轨迹数据
dl_train = list(Generatedataset(copy.deepcopy(data), seq_length, batch_size).values())[0]
dl_train是一个列表：[dataset1, dataset2....] dataset为每台出租车的轨迹训练集
"""
def Getdata(filepath, nrows):
    res = defaultdict(list)
    data = pd.read_csv(filepath,
                       header=None, nrows=nrows, names=["taxi_id", "order_id", "time", "longitude", "latitude"]
                       )
    taxi_ids = sorted(list(set(data["taxi_id"].values)))

    for taxi_id in taxi_ids:
        taxi_data = list(data[data["taxi_id"] == taxi_id].groupby("order_id"))  # 出租车的订单信息
        for trajectory in taxi_data:
            res[taxi_id].append(trajectory[1].loc[:, ["latitude", "longitude"]].values)
    return res

def Generatedataset(data_, seq_length, batch_size):
    train_data = defaultdict(list)
    for key in sorted(data_.keys()):
        for index in range(len(data_[key])):
            data_[key][index], _ = minMaxNormalize(data_[key][index], True)
            train_data[key].append(GenerateTrainData(data_[key][index], seq_length))
        ds = ConcatDataset(train_data[key])  # 拼接
        dl_train_ = DataLoader(ds, batch_size=batch_size)
        train_data[key] = dl_train_
    return train_data


if __name__ == "__main__":
    data = Getdata("D:\dataset\gps_01-10\gps_20161101", nrows=8000)
    print(data.keys())
    seq_length = 10
    batch_size = 32
    dl_train = list(Generatedataset(copy.deepcopy(data), seq_length, batch_size).values())
    dl_trains = list(Generatedataset(copy.deepcopy(data), seq_length, batch_size).values())[0]


