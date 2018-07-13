'''
Create on June 28, 2018

@author: ning
'''
import neural_net as NN
import numpy as np
import os
import gzip
import struct

def read_file(label_path, image_path):
    with gzip.open(label_path) as flab:
        magic, num = struct.unpack('>II', flab.read(8))       
        label = np.fromstring(flab.read(), dtype=np.int8)   
    with gzip.open(image_path, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack('>IIII', fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(num, rows, cols)
    return label, image

def load_minist_data():
    file_path='E:/projects/python/alexnet/src/MNIST_data'
    train_label_path = os.path.join(file_path, 'train-labels-idx1-ubyte.gz')
    train_image_path = os.path.join(file_path, 'train-images-idx3-ubyte.gz')
    test_label_path = os.path.join(file_path, 't10k-labels-idx1-ubyte.gz')
    test_image_path = os.path.join(file_path, 't10k-images-idx3-ubyte.gz')
    train_label, train_image = read_file(train_label_path, train_image_path)
    test_label, test_image = read_file(test_label_path, test_image_path)
    return train_image, train_label, test_image, test_label
# 加载训练数据
train_image, train_label, test_image, test_label = load_minist_data()
train_num = train_image.shape[0]
test_num = test_image.shape[0]
out_num = np.max(train_label) + 1
# 数据预处理:特征采用Min-Max normalization方法处理,输出标签转换为向量方式.
train_image = train_image.reshape(train_num, 28*28) / 255 - 0.5
train_sparse_label = np.zeros((train_num, out_num))
train_sparse_label[np.arange(0, train_num), train_label] = 1
test_image = test_image.reshape(test_num, 28*28) / 255 - 0.5
test_sparse_label = np.zeros((test_num, out_num))
test_sparse_label[np.arange(0, test_num), test_label] = 1



select = 1
if select > 0:
    # 开始训练
    a = NN.NeuralNet(hidden_layers=(100,), batch_size=200, learning_rate=1e-3,
                     max_iter=10, tol=1e-4, alpha=1e-5,
                     activation='relu', solver='adam')
    a.fit(train_image, train_sparse_label)
    y = a.predict(test_image)
    print('accuracy:', 1 - np.sum(np.abs(y - test_sparse_label)) / 2 / test_sparse_label.shape[0])
else:
    # sklearn中的MLP进行对比
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=30, alpha=1e-4,
    solver='adam', verbose=10, tol=1e-4, random_state=1,
    learning_rate_init=.001)
    mlp.fit(train_image, train_label)
    y = mlp.predict(test_image)
    print(1-np.sum(np.abs(y-test_label))/2/test_sparse_label.shape[0])

