from utils.fedavg import fedavg

class ServerTypeException(Exception):

    def __str__(self):
        print("类型错误，不是局部服务器")


class Server(object):
    # 服务器的父类
    def __init__(self, name):
        # 设置服务器的类型
        self.name = name

    def broadcast(self):
        # 广播参数
        pass

    def aggregate(self, weights):
        # 聚合参数
        pass

class LocalServer(Server):
    # 局部服务器类
    def __init__(self, name, net, mcsList):
        """
        :param name: 服务器的名字
        :param net: 神经网络模型
        :param mcsList: MCS列表
        """
        super(LocalServer, self).__init__(name)
        self.mcsList = mcsList  # MCS列表
        self.net = net  # 公共模型

    def broadcast(self):
        for mcs in self.mcsList:
            mcs.receiveModel(self.net)

    def aggregate(self, weights):
        weights = weights
        keys = self.net.state_dict().keys()
        parametersList = []
        for i in self.mcsList:
            parametersList.append(list(i.net.parameters()))
        paraAvg = fedavg(weights, parametersList, keys)  # 聚合参数
        self.net.load_state_dict(paraAvg)  # 更新模型参数

    def starttrain(self, Epochs):
        #开启训练

        #1. 广播模型

        #2. 训练MCS

        #3. 聚合参数


        pass

class GlobalServer(Server):
    # 全局服务器
    def __init__(self, name):
        super(GlobalServer, self).__init__(name)
        self.localServerList = []  # 区域服务器列表

    def addLocalServer(self, localserver):
        # 添加区域服务器
        if localserver.isinstance(LocalServer):
            self.localServerList.append(localserver)
        else:
            raise ServerTypeException
