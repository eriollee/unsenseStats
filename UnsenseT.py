import seaborn as sns
import numpy as np
import pandas as pd
from numpy.random import randn
import matplotlib.pyplot as plt
from sklearn import (manifold, decomposition, random_projection)
from numpy import *;
from sklearn import preprocessing
from scipy import stats


def loadDataSet(fileName,delim=','):
    # 打开文件
    fr=open(fileName);
    """
>>> line0=fr.readlines();
>>> type(line0)
<class 'list'>
>>> line0[0]
'10.235186\t11.321997\n'
    """
    stringArr=[line.strip().split(delim) for line in fr.readlines()];
    # map函数作用于给定序列的每一个元素,并用一个列表来提供返回值
    datArr=[list(map(lambda x:float(x),line)) for line in stringArr];
    dataMat=mat(datArr);
    return dataMat;
dataMat = loadDataSet('D:/require/shake_All.csv');



df=pd.DataFrame(data=dataMat)
moving_avg = pd.rolling_mean(df,5)
moving_avg = moving_avg.drop([0,1,2,3])
#X_pca = mat(X_pca)
#X_pcan=preprocessing.normalize(X_pca,norm='l2')
X_pca= decomposition.TruncatedSVD(n_components=1).fit_transform(moving_avg.T)
df=pd.DataFrame(data=X_pca)
df.T.to_csv('D:/require/shake_All2.csv')
print(df);


# style set 这里只是一些简单的style设置
sns.set_palette('deep', desat=.6)
sns.set_context(rc={'figure.figsize': (8, 5) } )
# np.random.seed(1425)



# df= pd.read_csv('D:/require/python/jpn2.csv')


# a1 = df['A']
# b1 = df['B']
#
# for i in range(0, len(df), 5) :
#     print(i)
#     a2 = np.array(a1)[0:i]
#     b2 = np.array(b1)[0:i]
#     with sns.axes_style("dark"):
#         sns.jointplot(a2, b2, kind="kde", size=20).plot_joint(plt.scatter, color="r")
#     plt.savefig("sns" + str(i) + ".png")  # save fig file
#
# a2 = np.array(a1)
# b2 = np.array(b1)
# with sns.axes_style("dark"):
#     sns.jointplot(a2, b2, kind="kde", size=20).plot_joint(plt.scatter, color="r")
# plt.savefig("sns" + str(len(a2)) + ".png")  # save fig file

#sns.plt.show()
