import numpy as np
from pandas import read_csv, DataFrame
import pandas as pd


path = 'D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/'


dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')





tipo = '.csv'  


def ler_base_componetes(path, tipo, nome):
    file = path + nome + tipo
    compomentes = read_csv(file)
    compomentes = compomentes.set_index('codigo')
    print(compomentes.shape)
    return compomentes


def ler_base_fechamento(path, tipo, nome):
    nome = '01_fechamento'
    file = path + nome + tipo
    fechamento = read_csv(file,sep = ';')
    fechamento = fechamento.set_index('codigo')
    datas = fechamento.columns.values.tolist()
    tikers = fechamento.index.values.tolist()
    values = fechamento.values
    fechamento = DataFrame(values, index=tikers, columns=datas, dtype = 'float64')
    fechamento = fechamento.stack()


    nome = '09_lpa'
    file = path + nome + tipo
    lpa = read_csv(file,sep = ';')
    lpa = lpa.set_index('codigo')
    
    datas = lpa.columns.values.tolist()
    tikers = lpa.index.values.tolist()
    values = lpa.values
    
    lpa = DataFrame(values, index=tikers, columns=datas, dtype = 'float64')
    lpa = lpa.stack()




    lpa = lpa.unstack()
    

    dados_ibov = fechamento.join(lpa, how='outer')








path             = 'D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/'
tipo             = '.csv'  
    
# carrega a base de componentes do IBOVESPA
nome = '00_componentes'
compomentes = ler_base_componetes(path, tipo, nome)

# carrega todo o histórico de preços de fechamento de todas as ações disponíveis do IBOVESPA
nome = '01_fechamento'
datas, fechamento = ler_base_fechamento(path, tipo, nome)

nome = '09_lpa'













from pandas import read_csv, DataFrame



lpa = read_csv(file,sep = ',')
path             = 'D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/'
tipo             = '.txt'  
nome             = 'lpa'
file = path + nome + tipo
lpa = read_csv(file,sep = ',')
data = lpa.iloc[:,1]
codigo = lpa.iloc[:,0]
lpa = lpa.iloc[:,2]
base_lpa = pd.DataFrame({'data': pd.to_datetime(data),
                         'codigo': codigo,
                         'lpa': lpa}, columns = ['data', 'codigo', 'lpa'])
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
path             = 'D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/'
tipo             = '.txt'  
nome             = 'fechamento'
file = path + nome + tipo
fechamento = read_csv(file,sep = ',')

data = fechamento.iloc[:,1]
codigo = fechamento.iloc[:,0]
fechamento = fechamento.iloc[:,2]
base_fechamento = pd.DataFrame({'data': pd.to_datetime(data),
                                'codigo': codigo,
                                'fechamento': fechamento}, columns = ['data', 'codigo', 'fechamento'])
base_total = pd.merge(base_fechamento, base_lpa, how = 'right', on = ['key1', 'key2'])
base_total = pd.merge(base_fechamento, base_lpa, how = 'right', on = ['data', 'codigo'])
base_total = pd.merge(base_fechamento, base_lpa, how = 'outer', on = ['data', 'codigo'])
















path = 'D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/'
tipo = '.csv'  
nome = '01_fechamento4'
file = path + nome + tipo
fechamento = read_csv(file,sep = ';')
fechamento = fechamento.set_index('codigo')
datas = fechamento.columns.values.tolist()
tikers = fechamento.index.values.tolist()
values = fechamento.values
fechamento = DataFrame(values, index=tikers, columns=datas, dtype = 'float64')
fechamento = fechamento.stack()
fechamento.to_csv('D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/fechamento_v2.txt', sep=',', index=True)    



















