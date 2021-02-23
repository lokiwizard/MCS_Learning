#传入需要预测的轨迹长度，测试集的路径，模型
def predict_trajectory(predict_length, data_length, test_path,model, classification="geolife"):
    if(classification == "geolife"):
        predict_list = []#存放每次预测的结果
        test_data = data_process(classification,test_path).values
        true_list = copy.deepcopy(test_data[data_length:data_length+predict_length])#存放真实数据
        for i in range(predict_length):
            x = copy.deepcopy(test_data[i:data_length+i])
            x,normal = NormalizeMult(x,True)
            x = x.reshape(1,data_length,2)
            y_hat = model.predict(x)
            y_hat = y_hat.reshape(y_hat.shape[1])
            y_hat = reshape_y_hat(y_hat, 2)
            y_hat = FNormalizeMult(y_hat, normal)
            predict_list.extend(y_hat.tolist())
    if(classification == "taxi"):
        predict_list = []#存放每次预测的结果
        test_data = data_process(classification,test_path)
        test_data = np.array(trajectory_compress(test_data.values))
        true_list = copy.deepcopy(test_data[data_length:data_length+predict_length])#存放真实数据
        for i in range(predict_length):
            x = copy.deepcopy(test_data[i:data_length+i])
            x,normal = NormalizeMult(x,True)
            x = x.reshape(1,data_length,2)
            y_hat = model.predict(x)
            y_hat = y_hat.reshape(y_hat.shape[1])
            y_hat = reshape_y_hat(y_hat, 2)
            y_hat = FNormalizeMult(y_hat, normal)
            predict_list.extend(y_hat.tolist())
            
    return true_list,predict_list
