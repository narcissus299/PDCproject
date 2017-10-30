from pynn import pynn
from pynnmp import pynnmp
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

digits = datasets.load_digits()
images,labels = digits.images, digits.target

im =plt.imshow(images[1])

n = len(images)
images = [np.reshape(i,(-1,1))/16 for i in images]
labels = [convertToVec(l) for l in labels]

train_images = images[:n/2]
train_labels = labels[:n/2]

test_images = images[n/2:]
test_labels = labels[n/2:]

plt.plot(train_images[0].reshape(8,8))

nn = pynnmp([64,30,10])

nn.train(train_images,train_labels, epochs=200)

acc=0.0
for img,lab in zip(test_images,test_labels):
	prediction = nn.predict(img)
	if converToLabel(prediction) == converToLabel(lab): acc += 1
	print converToLabel(prediction),converToLabel(lab)

print("\n")
print( "Accuracy : " + str(acc/len(test_labels)) )
