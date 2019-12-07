import numpy as np
from pandas import read_csv, DataFrame
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import SGD, RMSprop
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, LearningRateScheduler
import heapq as hq
import calendar
import time
import pandas as pd
import tensorflow as tf



def lstm_model_2(num_epochs,batch_size,lstm_neurons,dense_neurons,shape1,shape2) -> Model:
    
    # design network
    model = Sequential()
    model.add(LSTM(lstm_neurons, input_shape=(shape1, shape2),return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(lstm_neurons, input_shape=(shape1, shape2)))
    model.add(BatchNormalization())
    model.add(Dense(dense_neurons*lstm_neurons))
    model.add(BatchNormalization())
    model.add(Dense(dense_neurons))    
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.10, decay=0.019, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    model.summary()
    return model






def lstm_model(x_treino,y_treino,x_trade,y_trade,num_epochs,batch_size,lstm_neurons,dense_neurons,init_lr,pw,val_split,salvar):

    # design network
    model = Sequential()
    model.add(LSTM(lstm_neurons, input_shape=(x_treino.shape[1], x_treino.shape[2])))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    sgd = SGD(lr = 0.0, momentum = 0.9, decay = 0.0, nesterov = False)
    model.compile(loss='binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    cbks = [LearningRateScheduler(lambda x: 1. / (1. + x))]
#    lrate = LearningRateScheduler(poly_decay(num_epochs,init_lr,pw))
#    callbacks_list = [lrate]
    model.summary()
    
    # fit network
    history = model.fit(x_treino,
                        y_treino, 
                        validation_split = val_split,
                        epochs = num_epochs,
                        batch_size = batch_size,
                        callbacks = cbks,
                        verbose = 1,
                        shuffle = False)


    name = "D:/Users/felip/Documents/07. FEA/Dissertacao/codigos/modelos/model_lstm_" + str(lstm_neurons) + str(calendar.timegm(time.gmtime()))
    modeljson = name + "_retornos.json"
    modelhdf5 = name + "_retornos.h5"
    if salvar == 1:        
        # serialize model to JSON
        model_json = model.to_json()
        with open(modeljson, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(modelhdf5)
        print("Saved model to disk")


    # plot the training loss and accuracy
    H = history.history
    N = np.arange(0, len(H["loss"]))
    
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22
    
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title
    
    plt.figure()
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.plot(N, H["loss"], label="train_loss")
    plt.plot(N, H["val_loss"], label="valid_loss")
    plt.plot(N, H["acc"], label="train_acc")
    plt.plot(N, H["val_acc"], label="valid_acc")
    plt.title("LSTM on IBOV prediction task")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss / Accuracy")
    plt.legend(frameon=False)
    
    # evaluate performance on the trade period
    score_treino = model.evaluate(x_treino, y_treino, batch_size = batch_size,verbose = 1)
    score_trade  = model.evaluate(x_trade, y_trade, batch_size = batch_size, verbose = 1)
    print('Loss Treino.........:', score_treino[0])
    print('Loss Trade..........:', score_trade[0])
    print('Accuracy Treino.....:', score_treino[1])
    print('Accuracy Trade......:', score_trade[1])

    # return history and model
    return history, model, score_treino, score_trade, modeljson, modelhdf5










def poly_decay(epoch, init_lr, pw):
	# initialize the maximum number of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = num_epochs
	baseLR = init_lr
	power = pw
	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
	# return the new learning rate
	return alpha

























nome = '01_fechamento'
file = path + nome + tipo
fechamento = read_csv(file,sep = ';')
fechamento = fechamento.set_index('codigo')
datas = fechamento.columns.values.tolist()
tikers = fechamento.index.values.tolist()
values = fechamento.values
fechamento = DataFrame(values, index=tikers, columns=datas, dtype = 'float64')
fechamento = fechamento.stack()
path = 'D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/'
tipo = '.csv'  
nome = '01_fechamento'
file = path + nome + tipo
fechamento = read_csv(file,sep = ';')
fechamento = fechamento.set_index('codigo')
datas = fechamento.columns.values.tolist()
tikers = fechamento.index.values.tolist()
values = fechamento.values
fechamento = DataFrame(values, index=tikers, columns=datas, dtype = 'float64')
fechamento = fechamento.stack()
fechamento.to_csv('D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/fechamento.txt', sep=',', index=True)    
path = 'D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/'
tipo = '.csv'  
nome = '01_fechamento2'
file = path + nome + tipo
fechamento = read_csv(file,sep = ';')
fechamento = fechamento.set_index('codigo')
datas = fechamento.columns.values.tolist()
tikers = fechamento.index.values.tolist()
values = fechamento.values
fechamento = DataFrame(values, index=tikers, columns=datas, dtype = 'float64')
fechamento = fechamento.stack()
fechamento.to_csv('D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/fechamento.txt', sep=',', index=True)    

path             = 'D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/'
tipo             = '.txt'  
nome             = 'fechamento'
file = path + nome + tipo
fechamento = read_csv(file,sep = ',')
base_fechamento = pd.DataFrame({'data': pd.to_datetime(data),
                         'codigo': codigo,
                         'fechamento': fechamento}, columns = ['data', 'codigo', 'fechamento'])







base_total = pd.merge(base_fechamento, base_lpa, how = 'right', on = ['key1', 'key2'])
base_total = pd.merge(base_fechamento, base_lpa, how = 'right', on = ['data', 'codigo'])
base_total = pd.merge(base_fechamento, base_lpa, how = 'outer', on = ['data', 'codigo'])










# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()



model.add(LSTM(lstm_neurons,
               input_shape = (shape1, shape2), 
               activation='relu', 
               recurrent_activation='linear', 
               kernel_regularizer=regularizers.l2(0.01),
               recurrent_regularizer=regularizers.l2(0.01), 
               bias_regularizer=regularizers.l2(0.01), 
               activity_regularizer=regularizers.l2(0.01),
               stateful=True))
















# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_days*n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)











if reset_state == 1:    
    for i in range(n_epoch):
        print("Epoch ", i, "/", n_epoch)
        history = model.fit(train_X, train_y, epochs = 1, batch_size = n_batch, verbose = 1, shuffle = False)
        performance[i,0] = np.max(history.history["loss"])
        performance[i,1] = np.max(history.history["acc"])
        performance[i,2] = np.max(history.history["precision"])
        performance[i,3] = np.max(history.history["recall"])
        performance[i,4] = np.max(history.history["fmeasure"])
#        model.reset_states()
else:
    history = model.fit(train_X, train_y, epochs = n_epoch, batch_size = n_batch, verbose = 1, shuffle = False)
    performance[:,0] = history.history["loss"]
    performance[:,1] = history.history["acc"]
    performance[:,2] = history.history["precision"]
    performance[:,3] = history.history["recall"]
    performance[:,4] = history.history["fmeasure"]
















#    # análise descritiva da base de dados batch total - pré-MinMaxScaler
#    dataset_ret = DataFrame(base_batch_total)
#    plt.rcParams['figure.figsize'] = (12, 18)
#    print(dataset_ret.shape)
#    print(dataset_ret.head(20))
#    print(dataset_ret.describe())
#    # histograms
#    dataset_ret.iloc[:,:20].hist()
#    plt.show()
#    # scatter plot matrix
#    from pandas.plotting import scatter_matrix
#    scatter_matrix(dataset_ret.iloc[:,0:7])
#    plt.show()
    
    
    # seprada em teste e validação, escalona e reconstrói a base_batch_total
    aux = base_batch_total.reset_index(level=['data','codigo'])
    data_codigo_y = aux[['data','codigo','y']]
    colnames = list(aux)
    colnames = colnames[2:13]

    treino = base_batch_total.loc[dia_inicio_treino:dia_fim_treino, :]
    teste = base_batch_total.loc[dia_inicio_teste:dia_fim_teste, :]    
    treino_x = treino.iloc[:,0:treino.shape[1]-1]
    teste_x = teste.iloc[:,0:teste.shape[1]-1]

    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_x.fit(treino_x)
    treino_x = scaler_x.transform(treino_x)
    teste_x = scaler_x.transform(teste_x)
    
    base_x = np.append(treino_x, teste_x, axis=0)
    base_x.shape
    base_batch_total.shape
    
    mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
    
    # análise descritiva da base de dados batch total - pré-MinMaxScaler
    dataset_ret = DataFrame(treino_x)
    plt.rcParams['figure.figsize'] = (12, 18)
    print(dataset_ret.shape)
    print(dataset_ret.head(20))
    print(dataset_ret.describe())
    # histograms
    dataset_ret.iloc[:,:20].hist()
    plt.show()
    # scatter plot matrix
    from pandas.plotting import scatter_matrix
    scatter_matrix(dataset_ret.iloc[:,0:7])
    plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    
    



def dataprep(base_in, features, ret_target, time_steps, dia_inicio_treino, dia_fim_treino, dia_inicio_teste, dia_fim_teste):   

    base = base_in[features]
    base = base.set_index(['data'])    
    base['y'] = ((base['retorno'] > ret_target)).astype(np.int)    
    base = base.dropna(axis = 0, how = 'any')
    n_features = base.shape[1] - 1
    n_obs = time_steps * n_features
    #print('shape da base:', base.shape)

    qtde_treino = 0
    qtde_teste = 0
    #print("reframe each stock as supervised leaning, then stack all and reshape it")
    codigo_dist = base.iloc[:,0].drop_duplicates()
    for i in range(len(codigo_dist)):
        # get data from one stock
        aux = base.loc[base['codigo'] == codigo_dist.iloc[i]]
        aux = aux.drop(columns=['codigo'])

        # split into train and test sets        
        treino = aux.loc[dia_inicio_treino:dia_fim_treino, :]
        teste = aux.loc[dia_inicio_teste:dia_fim_teste, :]

        if treino.shape[0] != 0:

            scaler_x = MinMaxScaler(feature_range=(-1, 1))                        
            
            treino_x = treino.iloc[:,:treino.shape[1]-1].values
            treino_y = treino.iloc[:,-1].values.reshape(treino.shape[0],1)
            scaler_x.fit(treino_x)
            treino_x = scaler_x.transform(treino_x)
            treino = np.concatenate((treino_x, treino_y), axis = 1)            
            treino_reframed = series_to_supervised(treino, time_steps, 1)            
            treino_x = treino_reframed.iloc[:, :n_obs]
            treino_y = treino_reframed.iloc[:, -1]

            if i == 0:
                treino_x_total = treino_x
                treino_y_total = treino_y
            else:
                treino_x_total = treino_x_total.append(treino_x)
                treino_y_total = treino_y_total.append(treino_y)
            
            qtde_treino += 1
            
        if teste.shape[0] != 0:
            
            teste_x = teste.iloc[:,:teste.shape[1]-1].values
            teste_y = teste.iloc[:,-1].values.reshape(teste.shape[0],1)
            teste_x = scaler_x.transform(teste_x)
            teste = np.concatenate((teste_x, teste_y), axis = 1)
            teste_reframed = series_to_supervised(teste, time_steps, 1)
            teste_x = teste_reframed.iloc[:, :n_obs]
            teste_y = teste_reframed.iloc[:, -1]        
           
            if i == 0:
                teste_x_total = teste_x
                teste_y_total = teste_y
            else:
                teste_x_total = teste_x_total.append(teste_x)
                teste_y_total = teste_y_total.append(teste_y)
            
            qtde_teste += 1
        
        #print('shape de', codigo_dist.iloc[i], ':', treino_x.shape, treino_y.shape,teste_x.shape, teste_y.shape)
                    
    # reshape input to be 3D [samples (blocks of stocks), timesteps, features]
    treino_x = treino_x_total.values.reshape((treino_x_total.shape[0], time_steps, n_features))
    treino_y = treino_y_total.values.reshape((treino_y_total.shape[0], 1))
    teste_x = teste_x_total.values.reshape((teste_x_total.shape[0], time_steps, n_features))
    teste_y = teste_y_total.values.reshape((teste_y_total.shape[0], 1))
    

    
    print('shape final....:', treino_x.shape, treino_y.shape, teste_x.shape, teste_y.shape)
    print('# ações treino.:', qtde_treino)
    print('# ações teste..:', qtde_teste)
    print('# features.....:', n_features)
    
    return treino_x, treino_y, teste_x, teste_y






    

