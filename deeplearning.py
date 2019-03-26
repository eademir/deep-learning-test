import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def loadMNIST( prefix, folder ):
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile( folder + "/" + prefix + '-images-idx3-ubyte', dtype = 'ubyte' )
    magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
    data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )

    labels = np.fromfile( folder + "/" + prefix + '-labels-idx1-ubyte',
                          dtype = 'ubyte' )[2 * intType.itemsize:]

    return data, labels

trainingImages, trainingLabels = loadMNIST( "train", "data" )
testImages, testLabels = loadMNIST( "t10k", "data" )

trainingImages = trainingImages[:600,:,:]
testImages = testImages[:100,:,:]
trainingImages.shape

trainingLabels = trainingLabels.reshape(1,-1)
trainingLabels = trainingLabels[:,:600]
testLabels = testLabels.reshape(1,-1)
testLabels = testLabels[:,:100]

i=1
plt.imshow(trainingImages[i], cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
print(trainingLabels[:,i])

m_train = trainingLabels.shape[1]
m_test = testLabels.shape[1]
num_px = trainingImages[13].shape[0]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px)+")")

train_set_x_flatten = trainingImages.reshape(trainingImages.shape[0],-1).T
test_set_x_flatten = testImages.reshape(testImages.shape[0],-1).T


print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("trainingLabels shape: " + str(trainingLabels.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("testLabels shape: " + str(testLabels.shape))

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

trlabel = np.zeros(trainingLabels.shape)
trlabel[:,np.where(trainingLabels==1)[1]] = 1
trainingLabels = trlabel
tslabel = np.zeros(testLabels.shape)
tslabel[:,np.where(testLabels==1)[1]] = 1
testLabels = tslabel

def sigmoid(z):

    s = 1/(1+np.exp(-z))
    
    return s

def initialize_with_zeros(dim):

    w = np.random.randn(dim,1)*0.01
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

def propagate(w, b, X, Y):

    m = X.shape[1]
    m = np.int(m)
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T,X)+b)                                    # compute activation
    cost = -1/m*np.sum((Y*np.log(A))+(1-Y)*np.log(1-A))                                 # compute cost

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = np.divide((np.dot(X,(A-Y).T)),m)
    db = 1/m*(np.sum(A-Y)) 
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

    costs = []
    
    for i in range(num_iterations):
        
        grads, cost = propagate(w,b,X,Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w-(learning_rate*dw)
        b = b-(learning_rate*db)
        
        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
             Y_prediction[0, i] = 0
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

range(2,10)

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    
    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate,print_cost=True)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

d = model(train_set_x, trainingLabels, test_set_x, testLabels, num_iterations = 2000, learning_rate = 1, print_cost = False)

