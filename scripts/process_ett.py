import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler

# max min(0-1)
def norm(train, test):
    # 对训练集数据进行归一化，将特征缩放到[0,1]的范围
    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train)
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)
    return train_ret, test_ret

# downsample by 10
def downsample(data, labels, down_len):
    # 对数据进行降采样，将原始数据按照down_len进行降采样
    np_data = np.array(data)
    np_labels = np.array(labels)
    orig_len, col_num = np_data.shape
    down_time_len = orig_len // down_len

    # 对数据进行转置，方便按down_len进行降采样
    np_data = np_data.transpose()

    # 降采样过程，取每个down_len长度内的数据的中值作为降采样后的数据
    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)

    # 对标签进行降采样，如果down_len长度内存在异常，则该样本被标记为异常
    d_labels = np_labels[:down_time_len*down_len].reshape(-1, down_len)
    d_labels = np.round(np.max(d_labels, axis=1))

    # 将数据再次转置回原来的形状
    d_data = d_data.transpose()

    return d_data.tolist(), d_labels.tolist()

def main():
    # 从CSV文件中读取测试集和训练集数据
    test = pd.read_csv('data/ett/test.csv', index_col=0)
    train = pd.read_csv('data/ett/train.csv', index_col=0)

    # 使用reset_index方法将索引列重新设置为从1开始递增的整数
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    
    # # 去掉数据中的第一列（索引列）
    # test = test.iloc[:, 1:]
    # train = train.iloc[:, 1:]

    # # 填充缺失值，使用均值进行填充
    # train = train.fillna(train.mean())
    # test = test.fillna(test.mean())
    # train = train.fillna(0)
    # test = test.fillna(0)

    # # 去掉列名中的空格
    # train = train.rename(columns=lambda x: x.strip())
    # test = test.rename(columns=lambda x: x.strip())

    # 将训练集的标签列（OT）提取出来
    train_labels = train["OT"].copy()
    test_labels = test["OT"].copy()

    # # 去掉训练集和测试集中的标签列
    # train = train.drop(columns=['OT'])
    # test = test.drop(columns=['OT'])

    # 对数据进行归一化
    x_train, x_test = norm(train.values, test.values)

    # 将归一化后的数据更新到原始数据集中
    for i, col in enumerate(train.columns):
        train.loc[:, col] = x_train[:, i]
        test.loc[:, col] = x_test[:, i]

    # 对训练集和测试集进行降采样，down_len=10
    d_train_x, d_train_labels = train, train_labels
    d_test_x, d_test_labels = test, test_labels

    print(d_train_x)
    print(d_train_x)
    print(d_train_labels)
    print(d_test_labels)

    # 将降采样后的数据重新生成DataFrame，并加上标签列（attack）
    train_df = pd.DataFrame(d_train_x, columns=train.columns)
    test_df = pd.DataFrame(d_test_x, columns=test.columns)

    test_df['OT'] = d_test_labels
    train_df['OT'] = d_train_labels

    print(train_df['OT'])
    # # 去掉训练集中的前2160个样本，主要是为了配合数据处理中的滑动窗口操作
    # train_df = train_df.iloc[2160:]

    # 将处理后的训练集和测试集保存为CSV文件
    train_df.to_csv('data/ett1/train.csv')
    test_df.to_csv('data/ett1/test.csv')

    # 将训练集的特征列名保存到文本文件list.txt中
    f = open('data/ett1/list.txt', 'w')
    for col in train.columns:
        f.write(col+'\n')
    f.close()

if __name__ == '__main__':
    main()
