'''
Classifying (mini) MNIST with a feedforward neural network using Numpy only.
The code is flexible and can train any feedforward neural network architecture.
Author: LDC
'''

'''
Dependencies
'''
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf #only used to load MNIST


'''
Activation functions
'''
def softmax(x):
    ex = np.exp(x-np.mean(x)) #substract mean to prevent over/underflow
    sx = ex/np.sum(ex)
    if np.isnan(sx).any(): #in case of under/overflow
        x = np.float128(x)
        ex = np.exp(x - np.mean(x))  # substract mean to prevent over/underflow
        sx = ex / np.sum(ex)
        if np.isnan(sx).any():
            raise TypeError('Over/underflow in softmax function!')
    return sx

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    ex = np.exp(-x)
    if np.isnan(ex).any(): #in case of under/overflow
        x = np.float128(x)
        ex = np.exp(-x)
        if np.isnan(ex).any():
            raise TypeError('Over/underflow in sigmoid function!')
    return np.divide(1, ex+1)

def lancelu(x): #this is a relu with another threshold
    x=np.maximum(0, x)
    x=np.minimum(1000, x)
    return x


'''
Jacobian matrices of activation functions
'''

def softmax_jacobian(x):
    sx = softmax(x)
    dim = np.repeat(np.shape(sx), 2) #dimension of desired output
    result = np.diag(sx)-np.kron(sx,sx).reshape(dim)
    if np.isnan(result).any(): #in case of under/overflow
        x = np.float128(x) # to prevent over/underflow
        sx = softmax(x)
        dim = np.repeat(np.shape(sx), 2)  # dimension of desired output
        result = np.diag(sx) - np.kron(sx, sx).reshape(dim)
        if np.isnan(result).any():
            raise TypeError('Over/underflow in jacobian of softmax function!')
    return result

def relu_jacobian(x):
    return np.diag(x>=0)

def sigmoid_jacobian(x):
    ex = np.exp(-x)
    result = np.diag(np.divide(ex, (ex+1)**2))
    if np.isnan(result).any(): #in case of under/overflow
        x = np.float128(x)
        ex = np.exp(-x)
        result = np.diag(np.divide(ex, (ex + 1) ** 2))
        if np.isnan(result).any():
            raise TypeError('Over/underflow in jacobian of sigmoid function!')
    return result

def lancelu_jacobian(x):
    return np.diag((x >= 0) * (x <= 1000)) # * = elementwise multiplication

'''
Layer
'''
class Layer(object):

    '''
    Initialisation of layer
    '''
    def __init__(self, dim, activation = 'ReLU'):
        super(Layer, self).__init__() #contructs the object instance
        self.dim = dim    #dim is a numpy array with dim[0] the dimension of input and dim[1] the dimension of the output
        rand_w = np.random.normal(0, np.sqrt(2/np.sum(dim)),dim) #Xavier Glorot weight initialisation
        self.w = rand_w.reshape(np.flip(dim)) #weights initialised
        self.b =  np.zeros(np.prod(dim[1]))  # biases initialised to zero
        if activation not in ['ReLU', 'Softmax', 'Sigmoid', 'LanceLU']: # currently supported activation functions
            raise TypeError('Invalid activation function!')
        else:
            self.a = activation

    '''
    Methods to show attributes
    '''
    def show(self):
        return self.w, self.b, self.a

    def weights(self):
        return self.w

    def biases(self):
        return self.b

    def show_activation(self):
        return self.a

    def dimension(self):
        return self.dim
    '''
    Methods to set attributes
    '''
    def set_weights(self, w):
        if np.isnan(w).any():
            print(w)
            raise TypeError('Non number values in weight matrix!')
        if np.shape(w)== np.shape(self.w):
            self.w = w
        else:
            raise TypeError('Input weight matrix has wrong dimension!')

    def set_biases(self,b):
        if np.isnan(b).any():
            print(b)
            raise TypeError('Non number values in bias vector!')
        if np.shape(b) == np.shape(self.b):
            self.b = b
        else:
            raise TypeError('Input bias vector has wrong dimension!')

    '''
    Methods used in forward and backward pass
    '''
    def linear_map(self,x):
        return self.w @ x + self.b #W*x+b

    def activation(self,y):
        if self.a == 'ReLU':
            return relu(y)
        elif self.a == 'Softmax':
            return  softmax(y)
        elif self.a == 'Sigmoid':
            return sigmoid(y)
        elif self.a == 'LanceLU':
            return lancelu(y)
        else:
            raise TypeError('Invalid activation function!')

    def jacobian_activation(self, y):
        if self.a == 'ReLU':
            return relu_jacobian(y)
        elif self.a == 'Softmax':
            return  softmax_jacobian(y)
        elif self.a == 'Sigmoid':
            return sigmoid_jacobian(y)
        elif self.a == 'LanceLU':
            return lancelu_jacobian(y)
        else:
            raise TypeError('Invalid activation function!')


'''
Neural network
'''
class NeuralNetwork(object):

    def __init__(self, layers):
        super(NeuralNetwork, self).__init__() #constructs the object instance
        for i in range(len(layers)-1):  #checks that subsequent layer dimensions are compatible
            if layers[i].dimension()[1] != layers[i+1].dimension()[0]:
                raise TypeError('Layers have not been initialised to compatible dimensions: %s' % i,i+1)
        self.layers = layers #list of layers from first to last layer

    def forwardpass(self, x, l):
        #returns intermediate value of the neural network
        #if l= 2*number of layers -1, this returns the output
        #if l= 2*number of layers -2, this returns the value right before the final activation function
        #if l= 2*number of layers -3, this returns the value after the penultimate activation function
        #and so on
        out = self.layers[0].linear_map(x)
        i=1
        while i<=l:
            if i%2 == 1: #if i is odd
                out = self.layers[int(i/2)].activation(out)
            else:
                out = self.layers[int(i/2)].linear_map(out)
            i+=1
        return out

    def show(self):
        for i in range(len(self.layers)):
            print('Layer %s' % i)
            print(self.layers[i].show())

'''
Model
'''
class Model(object):

    def __init__(self, NN, loss = 'MSE', learning_rate = 1.e-2):
        super(Model, self).__init__()  # contructs the object instance
        self.NN = NN
        if loss not in ['MSE']:  # currently supported loss functions: 'MSE' mean squared loss
            raise TypeError('Invalid loss function!')
        else:
            self.loss = loss # set loss function
        #can add training method as attribute. For now just vanilla gradient descent
        if learning_rate <= 0:
            raise TypeError('Learning rate must be strictly positive!')
        self.learning_rate = learning_rate

    def train_model(self, data, label, epochs, batches, optim):
        #updates weights of neural network given dataset, loss function and optimisation method
        #data is the training data, input to the neural network
        #label are the target outputs from the training data
        #ALGORITHM outline:
        #split the data into batches of equal size
        #while not converged and within number of epochs
                # for each batch
                    # for each layer component starting with the latest bias vector, then latest weight matrix and so on.
                        # for each datapoint in batch
                            #forwardpass the datapoint through neural net
                            #compute gradients using backprop
                        #stochastic gradient descent

        if len(data) != len(label):
            raise TypeError('Number of inputs and labels do not coincide!')

        if batches <= 0 or not isinstance(batches, int):
            raise TypeError('Number of batches must be a positive integer')

        if len(data)<batches:
            batches = len(data)
            print(f'Number of batches set to {batches}')

        if optim == 'Adam':
            print('Network optimisation using Adam algorithm as shown in:')
            print('D.P. Kingma, J.L. Ba (2017). Adam: a method for stochastic optimisation')
        elif optim == 'SGD' and batches >1:
            print('Network optimisation using vanilla stochastic gradient descent')
        elif optim == 'SGD' and batches == 1:
            print('Network optimisation using standard gradient descent')
        else:
            raise TypeError('Optimisation algorithm not currently implemented')

        if np.size(label,1) != self.NN.layers[-1].dim[1]:
            raise TypeError('Output dimension of neural network differs from dimension of label data')

        '''
        Set up hyperparameters for Adam
        '''
        if optim == 'Adam': #Adam algorithm as shown in: 'D.P. Kingma, J.L. Ba (2017), Adam a method for stochastic optimisation'
            m = [0]*(2*len(self.NN.layers)) #initialise first moment vector
            v = [0]*(2*len(self.NN.layers)) #initialise second moment vector
            hm = [0]*(2*len(self.NN.layers)) #initialise bias corrected first moment vector
            hv = [0]*(2*len(self.NN.layers)) #initialise bias corrected second moment vector
            beta1= 0.9
            beta2=0.999
            epsilon = 10**(-8)
        t = 0  # timestep

        '''
        Training
        '''
        converged = False #not implemented but can setup a criterion for convergence in the while loop to stop training
        epoch = 1 #counter
        batch_size = int(len(data) / batches)

        while not converged and epoch <= epochs:

            print('=== Epoch === %s' % epoch)

            batch_order = np.random.permutation(len(data)) #shuffles the order in which we loop over the data
            sum_norm_grad = 0 #tracks the sum of gradients on each batch to identify vanishing gradients
            c_classified =0 # tracks correctly classified digits
            total_loss =0 # counts total loss function

            '''
            Loop over batches
            '''
            for b in range(batches): #for each batch
                t += 1 #update time for Adam

                grad = [None]*batch_size #initialise gradients for backprop
                grad2 = [None]*batch_size

                '''
                Loop over layer components (two components per layer)
                '''
                for l in range(2*len(self.NN.layers)-1, -1,-1):  # loop over layers backwards (for backprop)
                    # If l is odd this means we are computing the gradient for a bias update
                    # If l is even this means we are computing the gradient for a weight update
                    # This is why there are two values of l for each layer
                    # and we go backwards since we are backpropagating

                    '''
                    Loop over datapoints in batch
                    '''
                    for p in range(batch_size): #for each datapoint in batch

                        '''
                        Choose data point in batch
                        '''
                        index = batch_order[b * batch_size + p]  # index of batch datapoint in overall dataset
                        x = data[index]  # takes datapoint in batch
                        y = label[index]  # takes label of datapoint x


                        if l == 2*len(self.NN.layers)-1: #if last layer of neural net

                            '''
                            Output of neural net
                            '''
                            out = self.NN.forwardpass(x, l)  # forwardpass through the neural net
                            if np.isnan(out).any():  # tests if there are any non-numbers resulting from this computation
                                raise TypeError('Non-numerical value in output of neural net!')

                            '''
                            Gradient of loss function (squared error)
                            '''
                            grad[p] = 2 * (out - y) # gradient of loss function: squared error between output of neural net and target value

                            '''
                            Loss function
                            '''
                            total_loss += np.linalg.norm(out - y) ** 2  # track sum of squared errors

                            '''
                            Counts correctly classified data points
                            '''
                            if np.argmax(out) == np.argmax(y):
                                c_classified += 1  #add if digit correctly classified

                        '''
                        Forward-pass
                        '''
                        if l>0:
                            out = self.NN.forwardpass(x, l - 1)  # forwardpass through the neural net
                        else:
                            out = x

                        '''
                        Backward-pass
                        '''
                        if l%2==1: #if l is odd, i.e., if we are computing the gradient for a bias update
                            if l!= 2*len(self.NN.layers)-1:
                                grad[p] = grad[p] @ self.NN.layers[int(l/2)+1].weights() #multiply gradient by weight matrix
                            jac_activ = self.NN.layers[int(l/2)].jacobian_activation(out)
                            if np.isnan(jac_activ).any():
                                raise TypeError('Over/underflow in Jacobian computation')
                            grad[p] = grad[p] @ jac_activ #multiply gradient by jacobian of activation funciton
                            grad2[p]= grad[p]
                        else: #if l is even, i.e., if we are computing the gradient for a weight update
                            grad2[p]= grad2[p] @ np.tile(out,np.shape(grad2[p])).reshape([np.size(grad2[p]),np.size(out)])

                    '''
                    Update of Adam hyperparameters
                    '''
                    if optim == 'Adam':
                        m[l] = beta1 * m[l] + (1 - beta1) * sum(grad2) #Update biased first moment estimate
                        v[l] = beta2 * v[l] + (1 - beta2) * sum(grad2) * sum(grad2) #Update biased second raw moment estimate
                        hm[l]= m[l]/(1-beta1**t) #Compute bias-corrected first moment estimate
                        hv[l] = v[l] / (1 - beta2 ** t) #Compute bias-corrected second raw moment estimate
                        if np.sum(hv[l] <0)!=0: #checks for errors
                            raise TypeError("components of biased moment vector are negative")

                    '''
                    Stochastic gradient descent
                    '''
                    if l%2==1: #if l is odd
                        '''
                        SGD for biases
                        '''
                        if optim == 'Adam':
                            new_biases = self.NN.layers[int(l / 2)].biases() - self.learning_rate * hm[l] / (np.sqrt(hv[l]) + epsilon)
                        elif optim == 'SGD':
                            new_biases = self.NN.layers[int(l / 2)].biases() - self.learning_rate * sum(grad2)  # stochastic gradient descent step for biases
                        self.NN.layers[int(l / 2)].set_biases(new_biases)  # set new biases
                    else: #if l is even
                        '''
                        SGD for weights
                        '''
                        if optim == 'Adam':
                            new_weights = self.NN.layers[int(l / 2)].weights() - self.learning_rate * hm[l] / (np.sqrt(hv[l]) + epsilon)
                        elif optim == 'SGD':
                            new_weights = self.NN.layers[int(l / 2)].weights() - self.learning_rate * sum(grad2) # stochastic gradient descent step for weights
                        self.NN.layers[int(l / 2)].set_weights(new_weights)  # set new weights

                # scores L1 gradient norm over all batches
                sum_norm_grad += np.linalg.norm(sum(grad), ord=1)

            '''
            Print training and convergence statistics
            '''
            accuracy = np.round(c_classified / len(data), 4) #rounds the accuracy
            print('--- Total loss --- ~ %s' % total_loss)
            print(f'--- Correctly classified ---  {c_classified} / {len(data)}')
            print('--- Training accuracy --- ~ %s' % accuracy)
            print('--- Gradient norm (L1) --- %s' % sum_norm_grad) # reports sum of gradient norm over all batches (to check for vanishing gradients)
            #self.save() #save parameters
            epoch +=1

    def stats(self, data, label): # returns training statistics
        total_loss = 0
        c_classified = 0
        for i in range(len(data)):
            x = data[i]
            y = label[i]
            l=2*len(self.NN.layers)-1
            out = self.NN.forwardpass(x,l)
            total_loss += np.linalg.norm(out-y) ** 2 #Total squared error
            if np.argmax(out) == np.argmax(y):
                c_classified += 1
        accuracy = np.round(c_classified / len(data), 4)
        print('--- Total loss --- %s' % total_loss)
        print(f'--- Correctly classified --- {c_classified} / {len(data)}')
        print('--- Training accuracy --- %s' % accuracy)
        return total_loss, c_classified, accuracy

    def save(self): #saves model parameters so that training can be resumed later
        for l in range(len(self.NN.layers)):
            layer = self.NN.layers[l]
            w_name = 'weight %s' % l
            np.save(w_name, layer.weights())
            b_name = 'bias %s' % l
            np.save(b_name, layer.biases())
        for l in range(len(self.NN.layers)):
            a_name = 'activation %s' % l
            np.save(a_name, layer.show_activation())

'''
MAIN
'''
# Set random seed to get reproducible results
seed=1
np.random.seed(seed)

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Choose which digits to perform classification on
digits = range(5) #e.g., range(3) means we are taking only digits 0,1,2 in MNIST
indices_train = [] #keeps track of which digits in the training set have relevant images
indices_test = []
print(f'!=== CLASSIFICATION OF {len(digits)} DIGITS ===! {list(digits)} ')

# Process training data
x_train = x_train.reshape([60000,784]) #set each datapoint as a vector instead of a matrix
x_train = x_train/np.max(x_train) #normalise training data
y_t = y_train #create a copy
y_train =np.zeros([len(y_train), len(digits)])
for i in range(len(y_train)):
    if y_t[i] in digits:
        indices_train += [i]
        y_train[i, y_t[i]] = 1    #set each label as a vector of 10 entries (one for each digit) with one at the corresponding digit
x_train = x_train[indices_train, :] #select relevant digits in training data
y_train =y_train[indices_train,:] #select relevant digits in train labels

# Process test data
x_test = x_test.reshape([10000,784]) #set each datapoint as a vector instead of a matrix
x_test = x_test/np.max(x_test) #normalise training data
y_t = y_test #create a copy
y_test =np.zeros([len(y_test),len(digits)])
for i in range(len(y_test)):
    if y_t[i] in digits:
        indices_test += [i]
        y_test[i, y_t[i]] = 1
x_test = x_test[indices_test, :] #select relevant digits in test data
y_test =y_test[indices_test,:] #select relevant digits in test labels

# Neural network architecture
diml1 = np.array([784,100]) #dimension first layer (input, output)
diml2 = np.array([100,len(digits)])

# Create layers
l1 = Layer(diml1, 'ReLU') #first layer
l2 = Layer(diml2, 'Softmax') #choices of activation functions: ReLU, Sigmoid, Softmax

# Create neural network
layers = [l1,l2]
NN = NeuralNetwork(layers) #input a list of layers with compatible dimensions

# Show neural network weights
#NN.show()

#Specify learning rate
r= 10 **(-2)

# Create model
M = Model(NN, 'MSE', r) # MSE is the sum of squares loss function

#Set number of training epochs
epochs = 50

# Set number of batches for training
batches = int(len(x_train)/16)

#Optimisation scheme
optim= 'Adam' #Adam or SGD
#Note: optim= 'SGD' and batches=1 corresponds to gradient descent

# Train model
M.train_model(x_train, y_train, epochs, batches, optim)

# Convergence statistics on test set
print('!=== Convergence statistics on test set ===!')
M.stats(x_test, y_test)

# Save model
M.save()