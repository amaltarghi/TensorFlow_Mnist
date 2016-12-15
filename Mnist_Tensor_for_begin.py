# more informations https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import random
import numpy 
#parameters
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#activation
A = tf.nn.softmax(tf.matmul(x,w)+b)
#training
y_=tf.placeholder(tf.float32, [None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(A),reduction_indices=[1]))
#minimise the cross entropy using the descent gradient step gradient = 0.2
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
#initalize the defined variables
init = tf.initialize_all_variables()
# start tensorflow session
sess= tf.Session()
sess.run(init)
# training with our data 1000 iteration batch = 100
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
#Evaluating Our Model
prediction = tf.argmax(A,1)
correct_prediction = tf.equal(prediction, tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Finally, we ask for our accuracy on our test data
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#determine randomly an int references and element of training set
index=random.randint(0,len(mnist.test.images))

#take the image on the index and resize it 28*28 pixel
img= mnist.test.images[index]
img = img.reshape((28,28))

# the label associed to the image
label=mnist.test.labels[index]
num= numpy.where(label ==1) [0][0]

pred_num = sess.run(prediction,feed_dict={x:mnist.test.images,y_:mnist.test.labels})[index]

fig = plt.figure()
fig.suptitle (' write of a manuscrite {} (prédiction : {})'.format(num , pred_num),fontsize =14 , fontweight = 'bold' , color  = 'blue')
plt.imshow(img,cmap =cm.Greys)
plt.show()
