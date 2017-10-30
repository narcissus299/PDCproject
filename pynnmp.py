import numpy as np
import random
import pymp

class pynnmp:

	def __init__(self, size):
		self.number_of_layers = len(size)
		self.size = size
		self.bias = [np.random.randn(i,1) for i in size[1:]]
		self.weights = [np.random.randn(j,i) for i,j in zip(size[1:],size[:-1])]

	def sigmoid(self,z):
		return 1.0 / (1.0 + np.exp(-z))

	def d_sigmoid(self,z):
		return self.sigmoid(z) * (1-self.sigmoid(z))

	def feed_forward(self,x):
		pass

	def gradient_descent(self,training_data,training_label, minibatch_size, epochs, learning_rate, test_data, test_label):
		pass

	def back_prop(self, batch):

		x = batch[0]
		y = batch[1]

		dw = [np.zeros(w.shape) for w in self.weights ]
		db = [np.zeros(b.shape) for b in self.bias ]

		a = x
		A = []
		Fa = [x, ]

		for w,b in zip(self.weights,self.bias):
			a = w.T.dot(a) + b
			A.append(a)
			a = self.sigmoid(a)
			Fa.append(a)

		cost_grad = self.d_cost(Fa[-1],y)
		delta = cost_grad * self.d_sigmoid(A[-1])

		db[-1] = delta
		dw[-1] = Fa[-2].dot(delta.T)


		for i in range(2,self.number_of_layers):
			delta = (self.weights[-i+1].dot(delta)) * self.d_sigmoid(A[-i])
			db[-i] = delta
			dw[-i] = Fa[-i-1].dot(delta.T)

		return (dw,db)


	def d_cost(self,output,label):
		pass

	def loss(self,output,label):
		pass

	def train(self,data,label, minibatch_size, epochs, learning_rate ,test_data,test_label):
		pass

	def predict(self,data):
		return self.feed_forward(data)

	def test(self,data,label):
		pass
