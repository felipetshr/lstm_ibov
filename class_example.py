import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

path = 'D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/'


def plot_exemplo_semana(n, tam, x, y):
    nome = 'class_example_sup_learning_week.txt'
    file = path + nome
    fechamento = pd.read_csv(file,sep = ',')
    sample = fechamento.tail(n)
    # Function to map the colors as a list from the input list of x variables
    def pltcolor(lst):
        cols=[]
        for l in lst:
            if l==-1:
                cols.append('red')
            elif l==1:
                cols.append('blue')
            else:
                cols.append('green')
        return cols
    # Create the colors list using the function above
    cols=pltcolor(sample.classe)
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    ax.scatter(x=sample.ret_d_2,y=sample.ret_d_1,s = tam, c=cols, marker='o') #Pass on the list created by the function here
    pop_a = mpatches.Patch(color = 'red', label = 'resultado negativo')
    pop_b = mpatches.Patch(color = 'blue', label = 'resultado positivo')
    ax.legend(handles=[pop_a,pop_b], scatterpoints=1)
    fig.suptitle('Classificação do Resultado Semanal de ITUB4')
    plt.grid(False)
    plt.xlabel('retorno semanal r[t-2]')
    plt.ylabel('retorno semanal r[t-1]')
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)
    ax.axhline(y[0], color='gray', lw=1, linestyle = '--')
    ax.axhline(y[1], color='gray', lw=1, linestyle = '--')
    ax.axvline(x[0], color='gray', lw=1, linestyle = '--')
    ax.axvline(x[1], color='gray', lw=1, linestyle = '--')
    fig.savefig('ret_semanal_itub4.jpg')
    plt.show()
    

plot_exemplo_semana(n = 50, tam = 75, x = [0.01,0.075], y = [-0.0080,0.06])













