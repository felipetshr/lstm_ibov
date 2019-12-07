import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler



# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def plot_variaveis(base):
    plt.rcParams['figure.figsize'] = (12, 16)
    f, axarr = plt.subplots(base.shape[1], sharex=True)
    f.suptitle('Variáveis')
    for group in range(base.shape[1]):
        axarr[group].plot(base.iloc[:, group])
        axarr[group].set_title(base.columns[group], loc = 'right')
        f.subplots_adjust(hspace=0.5)
    for ax in axarr:
        ax.label_outer()
    plt.show()


def dataprep(base_in, features, ret_target, time_steps, dia_inicio_treino, dia_fim_treino, dia_inicio_teste, dia_fim_teste):   

    base = base_in.set_index(['data'])
    base = base[features]
    base['y'] = ((base['retorno'] > ret_target)).astype(np.int)    
    base = base.dropna(axis = 0, how = 'any')
    n_features = base.shape[1] - 1
    n_obs = time_steps * n_features
    print('shape da base:', base.shape)
#    base.head()
    qtde_treino = 0
    qtde_teste = 0
    print("reframe each stock as supervised leaning, then stack all and reshape it")
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
        
        print('shape de', codigo_dist.iloc[i], ':', treino_x.shape, treino_y.shape,teste_x.shape, teste_y.shape)
                    
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