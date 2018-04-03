import seaborn as sns
import numpy as np
import pandas as pd
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy import stats


# style set 这里只是一些简单的style设置
sns.set_palette('deep', desat=.6)
sns.set_context(rc={'figure.figsize': (8, 5) } )
np.random.seed(1425)
# figsize是常用的参数.

#x = stats.gamma(2).rvs(5000)
#y = stats.gamma(50).rvs(5000)
# with sns.axes_style("dark"):
#     sns.jointplot(x, y, kind="hex")

df= pd.read_csv('D:/require/python/58.csv')
    #D0=df['XAXIS']
    #D1=df['YAXIS']
    #D0N=preprocessing.normalize(D0,norm='l2')
    #D1N=preprocessing.normalize(D1,norm='l2')
# D1N=preprocessing.normalize(df,norm='l2')
#     #D0=D1N['XAXIS']
#     #D1=D1N['YAXIS']
# test=pd.DataFrame(data=D1N)
# test.to_csv('./result_norm.csv')
a1 = df[['1021.036379']]
b1 = df[['-2.281832628']]
a2 = np.array(a1)
b2 = np.array(b1)
with sns.axes_style("dark"):
    sns.jointplot(a2, b2, kind="kde", size=20).plot_joint(plt.scatter, color="r")
print("helloworld")
plt.show()
# sns.plt.show()