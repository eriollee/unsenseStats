import seaborn as sns
import numpy as np
import csv
import pandas as pd
from numpy.random import randn
import matplotlib.pyplot as plt
from sklearn import (manifold, decomposition, random_projection)
from numpy import *;
import utils.FileUtils as file
from sklearn import preprocessing
from scipy import stats
import os


def loadDataSet(fileName,delim=','):
    # 打开文件
    fr=open(fileName);
    # with open(fileName) as f:
    #     reader = csv.reader(f)

    stringArr=[line.strip().split(delim) for line in fr.readlines()];
    print(stringArr[1])
    # map函数作用于给定序列的每一个元素,并用一个列表来提供返回值
    datArr=[list(map(lambda x:float(x),line)) for line in stringArr];
    dataMat=mat(datArr);
    df = pd.DataFrame(data=dataMat)
    moving_avg = pd.rolling_mean(df, 5)
    moving_avg = moving_avg.drop([0, 1, 2, 3])
    X_pca = decomposition.TruncatedSVD(n_components=1).fit_transform(moving_avg.T)
    df = pd.DataFrame(data=X_pca)
    df.T.to_csv('D:/require/shake_All2.csv')
    print(df)
    # return dataMat;
# dataMat = loadDataSet('D:/require/shake_All.csv');

filename = 'D:/require/pca/shake_All_lichenming_20180402095400_y.csv'

def loadData(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        print(filename)
        stringArr = []
        title = []
        for row in reader:
            if reader.line_num != 1:
                # with open('D:/require/pca/temp.csv', 'w', newline='') as f:
                #     # 标头在这里传入，作为第一行数据
                #     writer = csv.writer(f)
                #     print(reader.line_num, row[1:])
                #     writer.writerow(row[1:])
                row_temp = row[1:len(row)-4]
                #print(row_temp)
                datArr = list(map(float,row_temp))
                stringArr.append(datArr)
            else:
                title = row
        df = pd.DataFrame(data=stringArr)
        moving_avg = pd.rolling_mean(df, 5)
        moving_avg = moving_avg.drop([0, 1, 2, 3])
        X_pca = decomposition.TruncatedSVD(n_components=1).fit_transform(moving_avg.T)
        df = pd.DataFrame(data=X_pca)
        insertName = pd.DataFrame([[filename.split('_')[2]]])
        insertName = insertName.append(df, ignore_index=True) #插入名字
        insertName = insertName.T
        insertName.columns = title[:len(title)-4]
        print(insertName)
        if os.path.exists('D:/require/shake_All2.csv'):
            insertName.to_csv('D:/require/shake_All2.csv', header=False, index=False , mode='a')#持续写入mode为a
        else:
            insertName.to_csv('D:/require/shake_All2.csv', header=True, index=False, mode='a')  # 持续写入mode为a

# loadData(filename)
L = file.file_name2('D:/require/pca')
for row in L:
    loadData(row)

# df=pd.DataFrame(data=dataMat)
# moving_avg = pd.rolling_mean(df,5)
# moving_avg = moving_avg.drop([0,1,2,3])
# X_pca= decomposition.TruncatedSVD(n_components=1).fit_transform(moving_avg.T)
# df=pd.DataFrame(data=X_pca)
# df.T.to_csv('D:/require/shake_All2.csv')
#print(df);


# style set 这里只是一些简单的style设置
# sns.set_palette('deep', desat=.6)
# sns.set_context(rc={'figure.figsize': (8, 5) } )
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
