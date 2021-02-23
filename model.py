"""
模型生成模块：
    
    train_X:features array([[[  ]..
    
                             [  ]]])
    
    train_Y:labels array([[  ]...
    
                          [  ]])
    
return model
"""
def create_model(train_X, train_Y):
    model = models.Sequential(
    [layers.LSTM(120,input_shape=(train_X.shape[1],train_X.shape[2]),return_sequences=True,name='lstm_1'),
     layers.Dropout(0.4,name='drop_1'),
     layers.LSTM(120,return_sequences=True,name='lstm_2'),
     layers.Dropout(0.4,name='drop_2'),
     layers.LSTM(120,return_sequences=False,name='lstm_3',),
     layers.Dropout(0.4,name='drop_3'),
     layers.Dense(train_Y.shape[1],activation='tanh',name='activation')
    ])
    model.compile(loss=['mse'],optimizer=tf.keras.optimizers.Adam(),metrics=['acc'])
    return model
