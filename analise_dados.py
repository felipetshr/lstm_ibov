dataset_ret = DataFrame(retornos_periodo_norm_t)
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


dataset_fec = DataFrame(log_fechamentos_periodo_std_t)
print(dataset_fec.shape)
print(dataset_fec.head(20))
print(dataset_fec.describe())
# histograms
dataset_fec.iloc[:,:20].hist()
plt.show()
# scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(dataset_fec.iloc[:,0:12])
plt.show()