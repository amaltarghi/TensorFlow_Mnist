# TensorFlow_Mnist
# Installation 
Before to start you must install the framework via pip (we use python)
<pre><code>
$ pip install tensorflow
</code></pre>
If There any problems try with this
<pre><code>
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc1-cp35-cp35m-linux_x86_64.whl
$ sudo pip3 install --upgrade $TF_BINARY_URL
</code></pre>
In order to test your installation:
<pre><code>
$ python3
Python 3.5.2 |Anaconda 4.2.0 (64-bit)| (default, Jul  2 2016, 17:53:06) 
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow
</code></pre>
# DATA
<pre><code>
$ mkdir MNIST_data
$ cd MNIST_data/
$ wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
</code></pre>
there 3 other files you will find them on the web site ( labels, test and test_labes)
it existe an other method to use this data, there are present on the sous-module:
tensorflow.examples.tutorial.mnist
<pre><code>
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import random
import numpy 

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True
#deterline randomly an int references and element of training set
index=random.randint(0,len(mnist.train.images))

#take the image on the index and resize it 28*28 pixel
img= mnist.train.images[index]
img = img.reshape((28,28))

# the label associed to the image
label=mnist.train.labels[index]
num= numpy.where(label ==1) [0][0]

fig = plt.figure()
fig.suptitle (' write of a manuscrite {}'.format(num),fontsize =14 , fontweight = 'bold' , color  = 'blue')
plt.imshow(img,cmap =cm.Greys)
plt.show()
</code></pre>


