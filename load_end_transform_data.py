from pandas import DataFrame, read_csv

path = 'D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/'

tipo = '.csv'  
nome = 'roe3'
file = path + nome + tipo
file = read_csv(file,sep = ';')
file = file.set_index('codigo')
datas = file.columns.values.tolist()
tikers = file.index.values.tolist()
values = file.values
file = DataFrame(values, index=tikers, columns=datas, dtype = 'float64')
file = file.stack()
file.to_csv(path + nome + '.txt', sep=',', index=True)



tipo = '.csv'  
nome = 'roic3'
file = path + nome + tipo
file = read_csv(file,sep = ';')
file = file.set_index('codigo')
datas = file.columns.values.tolist()
tikers = file.index.values.tolist()
values = file.values
file = DataFrame(values, index=tikers, columns=datas, dtype = 'float64')
file = file.stack()
file.to_csv(path + nome + '.txt', sep=',', index=True)    



tipo = '.csv'  
nome = 'pl3'
file = path + nome + tipo
file = read_csv(file,sep = ';')
file = file.set_index('codigo')
datas = file.columns.values.tolist()
tikers = file.index.values.tolist()
values = file.values
file = DataFrame(values, index=tikers, columns=datas, dtype = 'float64')
file = file.stack()
file.to_csv(path + nome + '.txt', sep=',', index=True)    


tipo = '.csv'  
nome = 'sharpe3'
file = path + nome + tipo
file = read_csv(file,sep = ';')
file = file.set_index('codigo')
datas = file.columns.values.tolist()
tikers = file.index.values.tolist()
values = file.values
file = DataFrame(values, index=tikers, columns=datas, dtype = 'float64')
file = file.stack()
file.to_csv(path + nome + '.txt', sep=',', index=True)


tipo = '.csv'  
nome = 'irf3'
file = path + nome + tipo
file = read_csv(file,sep = ';')
file = file.set_index('codigo')
datas = file.columns.values.tolist()
tikers = file.index.values.tolist()
values = file.values
file = DataFrame(values, index=tikers, columns=datas, dtype = 'float64')
file = file.stack()
file.to_csv(path + nome + '.txt', sep=',', index=True)

