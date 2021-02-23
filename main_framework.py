"""
本地模型训练模块：

第一步：先load预训练模型,没有预训练的模型参数可以不传，但是一定要调用一次

第二步：选择联邦训练则需要先广播模型，设置权重，训练几轮聚合一次参数（默认一轮聚合一次）
        选择集中化训练（串联）不需要设置以上三个值

第三步：开启联邦训练或者集中训练

"""


class MCS(object):
    def __init__(self,model):
        self.client_data = [] #存放再MCS中的数据集
        self.weights = []
        self.model = model
        
    def set_parallel_data(self, data, weights):
        self.client_data = data
        self.weights = weights
        
    def set_series_data(self, data):
        self.client_data = data

    def broadcast_model(self):
        model_list = [] #存放客户端模型的列表
        pretrain_model = self.model
        for client in self.client_data: #为每个客户端创建一个模型，并且设置预训练的参数
            model = create_model(client[0],client[1])
            model.set_weights(pretrain_model.get_weights())
            model_list.append(model)
        return model_list
    
    '''联邦平均算法'''
    def federate_average(self, parameters_list): #parameters=[C1,C2,C3....]参数集，权重可以自己进行设置
        res_list = []
        weight_sum = sum(self.weights)
        for i in range(len(parameters_list)):
            res_list.append(list(map(lambda x:x*self.weights[i],parameters_list[i])))
        result = []
        for i in range(len(res_list[i])):
            for j in range(len(res_list)):
                if j == 0:
                    total = np.zeros_like(res_list[j][i])
                total += res_list[j][i]
            result.append(total/weight_sum)       
        return result
    
    '''联邦训练'''
    def federated_train(self, epochs, model_list, train_epochs=1):#dataset=[[X1,Y1],[X2,Y2],[X3,Y3]....]
        dataset = self.client_data
        parameters_list = [] #存放每轮的参数
        metrics_list = []
        loss = []
        acc = []
        for i in range(epochs):
            print('parallel round_{r}:start!'.format(r=i))
            for i in range(len(model_list)):
                model_list[i].fit(dataset[i][0],dataset[i][1],epochs=train_epochs,verbose=0,batch_size=64)
                metrics_list.append([model_list[i].history.history['loss'][0],model_list[i].history.history['acc'][0]])
                parameters_list.append(model_list[i].get_weights())
            new_weights = self.federate_average(parameters_list) #每轮结束后计算联邦参数
            new_metrics = self.federate_average(metrics_list)
            loss.append(new_metrics[0])
            acc.append(new_metrics[1])
            print('loss:{l},acc:{a}'.format(l=new_metrics[0],a=new_metrics[1]))
            parameters_list = []
            metrics_list = []
            for model in model_list:
                model.set_weights(new_weights)
        #self.model.set_weights(new_weights)
        return new_weights,loss,acc
    
    '''初始化MCS模型的参数，可选：加载云服务器端的模型参数,否则使用默认参数。如果前面有串联/并联计算出的模型参数，可以传入'''
    def load_pre_model(self, pretrain_model_path=None, pre_train_result=None):
        if pretrain_model_path!=None:
            self.model = tf.keras.models.load_model(pretrain_model_path)
        if pre_train_result!=None:
            self.model.set_weights(pre_train_result)
        else:
            pass
        
    '''集中训练'''
    def centralize_train(self, outter_epochs=1, inner_epochs=1): #分为外部循环和内部循环，内部循环指的是每轮某个数据训练几次
        dataset = self.client_data
        pre_parameter = self.model.get_weights() #初始参数
        model_list = [] #存放数据集模型的列表
        loss = []
        acc = []
        for client in self.client_data: #为每个数据集创建一个模型
            model = create_model(client[0],client[1])
            model_list.append(model)
        for out in range(outter_epochs):
            print('series cycle: {o}: '.format(o=out))
            for i in range(len(self.client_data)):
                print('data {i} start'.format(i=i))
                model_list[i].set_weights(pre_parameter)
                history = model_list[i].fit(self.client_data[i][0],self.client_data[i][1],verbose=1,epochs=inner_epochs,batch_size=64)
                pre_parameter = model_list[i].get_weights()
                loss.extend(history.history.get('loss'))
                acc.extend(history.history.get('acc'))
                print('#'*30)
            print('*'*50)
        return pre_parameter,loss,acc
