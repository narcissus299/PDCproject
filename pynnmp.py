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
		for w,b in zip(self.weights,self.bias):
			x = self.sigmoid( w.T.dot(x) + b)
		return x

	def gradient_descent(self,training_data,training_label, minibatch_size, epochs, learning_rate, test_data, test_label):

		for i in range(epochs):
			train = zip(training_data,training_label)
			random.shuffle(train)
			mini_batches = [ train[k:k+minibatch_size] for k in xrange(0, len(train),minibatch_size)]

			for mini_batch in mini_batches:
				self.update_batch(mini_batch,learning_rate)

			'''Printing information (loss) about the current epoch'''
			if i%50==0: print("Epoch "+str(i)+"/"+str(epochs))

			if test_data and test_label:
				total_loss = np.zeros(self.bias[-1].shape)
				for data,lab in zip(test_data,test_label):
					prediction = self.predict(data)
					inv_loss = self.loss(prediction,lab)
					total_loss = np.add(total_loss,inv_loss)
				print np.mean(total_loss)

	def update_batch(self,mini_batch,learning_rate):

		dw = pymp.shared.list([np.zeros(w.shape) for w in self.weights ])
		db = pymp.shared.list([np.zeros(b.shape) for b in self.bias ])

		with pymp.Parallel(4) as p:
			for batch in p.iterate(mini_batch):			
				w_error, b_error = self.back_prop(batch)
				with p.lock:
					dw = [current_dw+w_change for current_dw,w_change in zip(dw,w_error)]
					db = [current_db+b_change for current_db,b_change in zip(db,b_error)]

		self.weights = [w - (learning_rate*change_w/len(mini_batch)) for w,change_w in zip(self.weights,dw)]
		self.bias = [b - (learning_rate*change_b/len(mini_batch)) for b,change_b in zip(self.bias,db)]

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
		return np.subtract(output, label)

	def loss(self,output,label):
		return 0.5 * ( np.subtract(label, output) ** 2 )

	def train(self,data,label, minibatch_size = 15, epochs= 1000, learning_rate= 10,test_data=None,test_label=None):
		self.gradient_descent(data,label,minibatch_size,epochs,learning_rate,test_data,test_label)

	def predict(self,data):
		return self.feed_forward(data)

	def test(self,data,label):
		pass
