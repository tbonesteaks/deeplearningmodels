import tensorflow as tf 
from tensorflow.keras.layers import Flatten
import matplotlib.pyplot as plt

#%matplotlib inline


#define kernel forward movement, softmax, a model for convenience, cross entropy function, and train steps
def forward (x):
    return tf.matmul(x,W) + b

def activate(x):
    return tf.nn.softmax(forward(x))

def model(x):
    x = flatten(x)
    return activate(x)

def cross_entropy(y_label, y_pred):
    #was getting errors in zero function so added 1.e-10
    return (-tf.reduce_sum(y_label * tf.math.log(y_pred + 1.e-10)))

#we need to uncomment one of the following to optimize the results, starting with the gradient descent optimizer
#scores are listed here for convenience (SGD ~ 84%, Nadam ~ 92%, FTRL ~ 92%, Adam 91.7%, AdamMax 89-90%)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.25)
#optimizer = tf.keras.optimizers.Nadam(learning_rate=.17,beta_2=.987,epsilon=1e-7)
#optimizer = tf.keras.optimizers.Adam(learning_rate=.15,epsilon=1e-7)
#optimizer = tf.keras.optimizers.Adamax(learning_rate=.17, epsilon=1e-7)
optimizer = tf.keras.optimizers.Ftrl(learning_rate=.17)

def train_step(x,y):
    with tf.GradientTape() as tape:
        current_loss = cross_entropy(y,model(x))
        grads = tape.gradient(current_loss , [W,b])
        optimizer.apply_gradients(zip(grads,[W,b]))
    return current_loss.numpy()

#end function definitions

#get the mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0


print("categorical labels")
print(y_train[0:5])

#one hot encode these
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

print("One hot encoded labels")
print(y_train[0:5])

print("Number of training examples: " , x_train.shape[0])
print("Number of test examples: " , x_test.shape[0])

#make the datasets and prepare batches
#batchsize
bb = 200

#datasets - this cuts up our datasets based on the size of the batches set with the bb variable
train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(bb)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(bb)

#Convert 2d images into 1D vectors with Flatten and output the difference to stdout
flatten = Flatten(dtype='float32')

print("Original training data shape:", x_train.shape)
print("Flattened training data shape:", flatten(x_train).shape)

#assign bias and weight tensors - tf.zeros puts a 0 into each place in the array prior to computation
W = tf.Variable(tf.zeros([784,10], tf.float32))
b = tf.Variable(tf.zeros([10], tf.float32))

#run and output a cross entropy calculation against all those zeros. 
print("Current loss for unoptimized model:")    
startloss = cross_entropy(y_train, model(x_train)).numpy()
print(startloss)

#we need some empty arrays, we can't append to them unless they are defined
loss_values = []
accuracies = []

#how many epochs shall we run?
epochs = 12
# FYI; that means 12 epochs * batches = to compute all the model steps in training. 
# bb = batch size, 60,000 photos are in the dataset / bb = batches 
# 60k/100 = 600 batches & 60k/200 = 300 batches.
# When bb = 100 you are running 7200 steps over 12 epochs. When bb = 200 you are running 3600 steps over 12 epochs. 
# This will take 30-60 mins without GPU parallelization.
# If you run tensorflow with GPU parallelization, set bb to about 40-70 and this takes under 5 minutes.

#setup a loop for each epoch, and then another to loop through the data in the datasets by batch
for i in range(epochs):
    m=0
    for x_train_batch , y_train_batch in train_ds:
        m+=1
        current_loss = train_step(x_train_batch, y_train_batch)
        if m%150==0: ##export a status report 1/2 way through each epoch (there are 600 batches, we call on 300 for bb=100 on 150 for bb=200)
            print("Model Training Status Report; in Epoch #", str(i), " Batch", str(m), " Loss is about ", str(current_loss) , ". We began with ", str(startloss) , " loss.")

        #collect epoch stats

        #loss appending
        current_loss = cross_entropy(y_train , model(x_train)).numpy()
        loss_values.append(current_loss)
        correct_prediction = tf.equal(tf.argmax(model(x_train), axis=1), tf.argmax(y_train, axis=1))

        #accuracy appending
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
        accuracies.append(accuracy)

        #output Epoch Stats
        print("End of Epoch #" , str(i) , "  Batch", str(m), "  Loss=", str(current_loss), " Accuracy=", str(accuracy))

# Even though we output those values to stdout to watch the network, we'll also use them in the graphs below.

# Here we'll compute a summary statistic of the accuracy of both the training and test data.
# This is being done to ensure we are not overfitting (would manifest as a large deviation between the two accuracy percentages.)
# Also another reason why setting the model was done, convenience. 

correct_prediction_train = tf.equal(tf.argmax(model(x_train), axis=1),tf.argmax(y_train, axis=1))
accuracy_train = tf.reduce_mean(tf.cast(correct_prediction_train,tf.float32)).numpy()

correct_prediction_test = tf.equal(tf.argmax(model(x_test), axis=1),tf.argmax(y_test, axis=1))
accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test,tf.float32)).numpy()

print("Training Dataset Accuracy: " , str(accuracy_train * 100), "%" )
print("Testing Dataset Accuracy: " , str(accuracy_test * 100), "%" )
accuracy_average = ( float(accuracy_train) + float(accuracy_test) )/2
print("Average of the Above: " , str(accuracy_average * 100), "%" )

# use matplotlib to output some graphs of the data
#comment out subplot lines, and uncomment plt.show() to separate graphs
plt.subplot(2,1,1)
plt.plot(loss_values,'-mo')
plt.title("Loss")
plt.xlabel("Batches Processed")
plt.ylabel("Loss")
# plt.show()

plt.subplot(2,1,2)
plt.plot(accuracies,'-yo')
plt.title("Accuracy")
plt.xlabel("Batches Processed")
plt.ylabel("Accuracy")
plt.show()

print("Model is at least" ,  str((accuracy_average*100)-80) , " points over 80% accurate.")

