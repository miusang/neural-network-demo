'''
Create on June 27, 2018

@author: ning
'''
import numpy as np
import math
import copy
# import matplotlib.pyplot as plt
from scipy.special import expit as logistic_sigmoid
import warnings

def softmax(x):
    '''输出层输出，计算多分类问题的概率
    '''
    x_shift = x - np.max(x, axis=1, keepdims=True)        
    exp_x = np.exp(x_shift)
    softmax_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return softmax_x

def sigmoid(x):
    '''激活函数，用于隐藏层输出
    '''
    #exp_x = np.exp(-x)
    #sigmoid_x = 1 / (1 + exp_x)
    sigmoid_x = logistic_sigmoid(x)
    return sigmoid_x

def relu(x):
    '''激活函数，用于隐藏层输出
    '''    
    relu_x = (np.abs(x) + x) / 2
    return relu_x


class NeuralNet(object):
    '''多层感知机分类器 Multi-layer Perceptron classifier.
    This model optimizes the log-loss function using stochastic gradient descent.

    Parameters
    ----------
    hiddem_layers : tuple, length = n_layers.
        The ith element represents the number of neurons in the ith hidden layer.

    activation : {'sigmoid', 'relu'}
        Activation function for the hidden layer.

    solver : {'sgd', 'adam'}

    batch_size : int
        Size of minibatches for stochastic optimizers.

    learning_rate : float

    max_iter : int
        Maximum number of iterations.

    alpha : float
        L2 penalty (regularization term) parameter.

    tol : float
        Tolerance for the optimization.
    '''
    def __init__(self, hidden_layers, batch_size,
                 activation, learning_rate, max_iter,
                 solver, tol, alpha):        

        hidden_layers = list(hidden_layers)
        self.hidden_layers = hidden_layers
        self.n_layers = len(hidden_layers)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = 1e-8
        self.solver = solver
        self.alpha = alpha
        
        # 激活函数(隐藏层)的选择        
        if activation == 'sigmoid':
            self.activation = sigmoid
        elif activation == 'relu':
            self.activation = relu

    def fit(self, X, y):
        '''Fit the model to data matrix X and target(s) y.

        Parameter
        ---------
        X : array-like or sparse matrix, shape (nsamples, nfeatures)
            The imput data.

        y : array-like, shape (nsamples, noutputs)
            The target values.

        Returns
        -------
        self : return trained neural-network model.
        '''
        self._fit(X, y)

    def predict(self, X):
        activations = []
        for i in range(self.n_layers):
            activations.append(np.empty((X.shape[0], self.hidden_layers[i]), np.float))
        activations.append(np.empty((self.batch_size, self.noutputs,), np.float))
        activations = self.__forward(X, activations)
        a = activations[self.n_layers]
        y = a / np.max(a, axis=1, keepdims=True)
        y[y < 1] = 0
        return y

    def _fit(self, X, y):
        X, y = self.__Check_X_y(X, y)
        nsamples, nfeatures = X.shape
        self.noutputs = y.shape[1]
        print("train samples num:%d,"%(nsamples), "features num:%d,"%(nfeatures), "output num:%d"%(self.noutputs))

        # 初始化参数：weights, bias, grad_weights, grad_bias
        # 备注：参数的初始化很重要，第一种初始化方式，其训练精度最高只有84%左右，而且训练周期比较长；
        #     第二种初始化方法，训练精度高，可达到97%以上，训练周期短，效率高。
        self.W = []
        self.b = []
        # self.W.append(np.random.randn(nfeatures, self.hidden_layers[0]) / np.sqrt(2.0 / nfeatures))
        # self.b.append(np.random.randn(self.hidden_layers[0]) / np.sqrt(2.0 / self.hidden_layers[0]))
        in_num = nfeatures
        out_num = self.hidden_layers[0]
        init_bound = np.sqrt(2.0 / (in_num + out_num))
        self.W.append(np.random.uniform(-init_bound, init_bound, (in_num, out_num)))
        self.b.append(np.random.uniform(-init_bound, init_bound, out_num))
        for i in range(self.n_layers - 1):
            in_num = self.hidden_layers[i]
            out_num = self.hidden_layers[i+1]
            # self.W.append(np.random.randn(in_num, out_num) / np.sqrt(2.0 / in_num))
            # self.b.append(np.random.randn(self.hidden_layers[i+1]) / np.sqrt(2.0 / self.hidden_layers[i+1]))
            init_bound = np.sqrt(2.0 / (in_num + out_num))
            self.W.append(np.random.uniform(-init_bound, init_bound, (in_num, out_num)))
            self.b.append(np.random.uniform(-init_bound, init_bound, out_num))
        # self.W.append(np.random.randn(self.hidden_layers[self.n_layers - 1],
        # self.noutputs) / np.sqrt(2.0 / self.hidden_layers[self.n_layers - 1]))
        # self.b.append(np.random.randn(self.noutputs) / np.sqrt(2.0 / self.noutputs))
        in_num = self.hidden_layers[self.n_layers - 1]
        out_num = self.noutputs
        init_bound = np.sqrt(2.0 / (in_num + out_num))
        self.W.append(np.random.uniform(-init_bound, init_bound, (in_num, out_num)))
        self.b.append(np.random.uniform(-init_bound, init_bound, out_num))
        # print('weights:', self.W)
        # print('bias:', self.b)
        
        # 每次激活函数值保存位置，计算梯度会用到
        activations = []
        for i in range(self.n_layers):
            activations.append(np.empty((self.batch_size, self.hidden_layers[i]), np.float))
        activations.append(np.empty((self.batch_size, self.noutputs,), np.float))

        # 梯度参数存放位置
        self.grad_W = copy.deepcopy(self.W)
        self.grad_b = copy.deepcopy(self.b)

        # 用来存放训练过程中，得到验证损失最小时的参数
        self.best_W = copy.deepcopy(self.W)
        self.best_b = copy.deepcopy(self.b)
        self.best_loss = np.inf
        self.loss_curve = []
        
        # 梯度优化方法的选择
        if self.solver == 'adam':
            # adam参数初始化(Adaptive Moment Estimation)
            self.iter_count = 0
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.mt_W = copy.deepcopy(self.W)
            self.mt_b = copy.deepcopy(self.b)
            self.vt_W = copy.deepcopy(self.W)
            self.vt_b = copy.deepcopy(self.b)
            for i in range(self.n_layers + 1):
                self.mt_W[i][:] = 0.0
                self.mt_b[i][:] = 0.0
                self.vt_W[i][:] = 0.0
                self.vt_b[i][:] = 0.0
        elif self.solver == 'sgd':
            # NAG参数初始化(nesterov accelerated gradient)
            self.momentum = 0.9
            self.velocities_W = copy.deepcopy(self.W)
            self.velocities_b = copy.deepcopy(self.b)
            for i in range(self.n_layers + 1):
                self.velocities_W[i][:] = 0.0
                self.velocities_b[i][:] = 0.0
        
        print('start training...')
        try:
            # epoch loop
            for epoch_count in range(self.max_iter):
                # 打乱数据
                shuffle_index = np.random.permutation(nsamples)
                shuffle_X = X[shuffle_index]
                shuffle_y = y[shuffle_index]
                # batch loop
                accumulated_loss = 0.0
                for i in range(0, nsamples - self.batch_size + 1, self.batch_size):
                    # 抽取批量数据
                    batch_X = shuffle_X[np.arange(i, i + self.batch_size)]
                    batch_y = shuffle_y[np.arange(i, i + self.batch_size)]
                    # print(index)
                    # print('batch_X:', batch_X[0])
                    # print('batch_y:', batch_y[0])
                
                    if self.solver == 'sgd':
                        for j in range(self.n_layers + 1):
                            self.W[j] += self.momentum * self.velocities_W[j] 
                            self.b[j] += self.momentum * self.velocities_b[j]                    

                        activations = self.__forward(batch_X, activations)
                        self.__backpro(batch_X, batch_y, activations)
                        self.__update_params()

                        activations = self.__forward(batch_X, activations)
                        batch_loss = self.__compute_loss(activations[self.n_layers], batch_y)
                        accumulated_loss += (batch_loss / self.batch_size)
                    elif self.solver == 'adam':
                        activations = self.__forward(batch_X, activations)
                        self.__backpro(batch_X, batch_y, activations)
                        self.__update_params()
                        
                        activations = self.__forward(batch_X, activations)
                        batch_loss = self.__compute_loss(activations[self.n_layers], batch_y)
                        accumulated_loss += (batch_loss / self.batch_size)

                # 计算本轮epoch损失，训练精度
                loss = accumulated_loss
                pre_y = self.predict(X)
                accuracy = 1 - np.sum(np.abs(y - pre_y)) / 2.0 / nsamples
                print('%dth-epoch-loss:'%(epoch_count), loss, 'accuracy:', accuracy)

                self.loss_curve.append(loss)
                self.__update_no_improvement_count()
                if self.no_improvement_count > 2:
                    print('Training loss did not improve more than tol=%f'
                          'for two consecutive epochs.' %self.tol)
                    self.W = self.best_W
                    self.b = self.best_b
                    break
                if epoch_count + 1 == self.max_iter:
                    warnings.warn('Stochastic Optimizer: Maximum iterations (%d) '
                          'reached and the optimization hasn\'t converged yet.'
                          % self.max_iter)
        except KeyboardInterrupt:
            warnings.warn('Training interrupted by user.')

    def __update_no_improvement_count(self):
        '''检查是否达到停止训练的条件
        '''
        if self.loss_curve[-1] > self.best_loss - self.tol:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0

        if self.loss_curve[-1] < self.best_loss:
            self.best_loss = self.loss_curve[-1]
            self.best_W = self.W.copy()
            self.best_b = self.b.copy()

    def check_grad(self):
        '''梯度检查，数值梯度(numerical gradient)和解析梯度(analytic gradient)
           进行对比，注意，要用相对误差进行计算     
        '''
        deta = 1e-5
        index = 0        
        self.W[index][1, 0] += deta
        activations = self.__forward(batch_X, activations)
        #print(activations[self.n_layers-1])
        #print(activations[self.n_layers])
        loss1 = self.__compute_loss(activations[self.n_layers], batch_y)
        print('loss1:', loss1)
        
        self.W[index][1, 0] -= 2.0 * deta
        activations = self.__forward(batch_X, activations)
        loss2 = self.__compute_loss(activations[self.n_layers], batch_y)
        print('loss2:', loss2)
        
        slope = (loss1 - loss2)/(2.0 *deta)
        print('数值梯度:', slope)
        self.W[index][1, 0] += deta
        print('解析梯度:', self.grad_W[index][1, 0])
        
    def __forward(self, X, activations):
        '''前向传播 Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.
        
        Parameters
        ----------
        X : array-like, shape (nsamples, nfeatures)

        activations : list, length = n_layers + 1
            The ith element of the list holds the values of the ith layer and the
            output layer.
        '''
        activations[0] = self.activation(np.dot(X, self.W[0]) + self.b[0])
        for i in range(self.n_layers - 1):
            activations[i+1] = self.activation(np.dot(activations[i], self.W[i+1]) + self.b[i+1])
        activations[self.n_layers] = softmax(np.dot(activations[self.n_layers - 1],
                                                    self.W[self.n_layers])+ self.b[self.n_layers])
        return activations

    def __backpro(self, X, y, activations):
        '''反向传播，计算参数梯度        
        '''
        # 计算梯度
        if self.activation == relu:
            # 隐藏层的激活函数relu
            a = activations[self.n_layers]
            de_da = -y / (a + self.epsilon) / self.batch_size
            de_dz = a * (de_da - np.sum(a * de_da, axis=1, keepdims=True))
            for i in range(self.n_layers):
                self.grad_W[self.n_layers - i] = np.dot(activations[self.n_layers - i - 1].T, de_dz)
                self.grad_b[self.n_layers - i] = np.sum(de_dz, axis=0)
                de_da = np.dot(de_dz, self.W[self.n_layers - i].T)
                da_dz = copy.deepcopy(activations[self.n_layers - i - 1])
                da_dz[da_dz > 0] = 1
                de_dz = de_da * da_dz
            self.grad_W[0] = np.dot(X.T, de_dz)
            self.grad_b[0] = np.sum(de_dz, axis=0)
        elif self.activation == sigmoid:
            # 隐藏层的激活函数sigmoid
            a = activations[self.n_layers]            
            de_da = -y / (a + self.epsilon) / self.batch_size
            de_dz = a * (de_da - np.sum(a * de_da, axis=1, keepdims=True))
            for i in range(self.n_layers):
                self.grad_W[self.n_layers - i] = np.dot(activations[self.n_layers - i - 1].T, de_dz)
                self.grad_b[self.n_layers - i] = np.sum(de_dz, axis=0)
                de_da = np.dot(de_dz, self.W[self.n_layers - i].T)
                a = activations[self.n_layers - i - 1]
                da_dz = a * (1 - a)
                de_dz = de_da * da_dz
            self.grad_W[0] = np.dot(X.T, de_dz)
            self.grad_b[0] = np.sum(de_dz, axis=0)
        # L2正则化梯度计算
        for i in range(self.n_layers + 1):
            self.grad_W[i] += self.alpha * self.W[i]

    def __update_params(self):
        '''参数更新
        '''
        # 1. 梯度下降法，一般不用
        # self.W[i] -= self.learning_rate * self.grad_W[i]
        # self.b[i] -= self.learning_rate * self.grad_b[i]
        if self.solver == 'sgd':
            # NAD梯度优化
            for i in range(self.n_layers + 1):                
                self.velocities_W[i] = self.momentum * self.velocities_W[i] - self.learning_rate * self.grad_W[i]
                self.velocities_b[i] = self.momentum * self.velocities_b[i] - self.learning_rate * self.grad_b[i]

                self.W[i] += self.velocities_W[i]
                self.b[i] += self.velocities_b[i]
        elif self.solver == 'adam':
            # adam梯度优化方法
            self.iter_count += 1
            for i in range(self.n_layers + 1):                
                self.mt_W[i] = self.beta1 * self.mt_W[i] + (1 - self.beta1) * self.grad_W[i]
                self.mt_b[i] = self.beta1 * self.mt_b[i] + (1 - self.beta1) * self.grad_b[i]
                self.vt_W[i] = self.beta2 * self.vt_W[i] + (1 - self.beta2) * (self.grad_W[i] ** 2)
                self.vt_b[i] = self.beta2 * self.vt_b[i] + (1 - self.beta2) * (self.grad_b[i] ** 2)

                learning_rate = (self.learning_rate * np.sqrt(1 - self.beta2 ** self.iter_count) /
                                 (1 - self.beta1 ** self.iter_count))
                self.W[i] += -learning_rate * self.mt_W[i] / (np.sqrt(self.vt_W[i]) + self.epsilon)
                self.b[i] += -learning_rate * self.mt_b[i] / (np.sqrt(self.vt_b[i]) + self.epsilon)
##                beta1_t = self.beta1 ** self.iter_count
##                beta2_t = self.beta2 ** self.iter_count
##
##                mt_W_esti = self.mt_W[i] / (1 - beta1_t)
##                mt_b_esti = self.mt_b[i] / (1 - beta1_t)
##                vt_W_esti = self.vt_W[i] / (1 - beta2_t)
##                vt_b_esti = self.vt_b[i] / (1 - beta2_t)
##            
##                self.W[i] += -self.learning_rate * mt_W_esti[i] / (np.sqrt(vt_W_esti) + self.epsilon)
##                self.b[i] += -self.learning_rate * mt_b_esti[i] / (np.sqrt(vt_b_esti) + self.epsilon)

    def __compute_loss(self, x, y):
        '''计算损失值

        Paramters
        ---------
        x : the output layer values, shape (nsamples, noutputs)

        Returns
        -------
        loss: float
        '''
        # 对数似然函数 loss = ylog(x + 1e-8)
        prob = np.clip(x, 1e-10, 1.0 - 1e-10)
        loss = -np.sum(y * np.log(prob))
        # L2正则化损失
        for i in range(self.n_layers + 1):
            loss += self.alpha * (np.sum(self.W[i] * self.W[i]) / 2.0)
        return loss
    
    def __Check_X_y(self, X, y):
        '''检查数据格式
        '''
        if not X.ndim == 2:
            nfeatures = X.shape[0]
            X = X.reshape(1, nfeatures)
        if not y.ndim == 2:
            noutputs = y.shape[0]
            y = y.reshape(1, noutputs)
        return X, y
    
