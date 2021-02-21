import numpy as np


####################################################################
## Code borrowed from https://github.com/cs231n/cs231n.github.io

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def state_dict(self,):
        return self.params

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
        an integer in the range 0 <= y[i] < C. This parameter is optional; if it
        is not passed then we only return scores, and if it is passed then we
        instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
        samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
        with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        #############################################################################
        # Forward pass, computing the class scores for the input.                   #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        H = X.dot(W1) + b1[None, :]
        ReLU_H = H.copy()
        ReLU_H[H < 0] = 0
        scores = ReLU_H.dot(W2) + b2[None, :]
        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # Compute the loss. This should include                                     #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar.                           #
        #############################################################################
        scores -= np.max(scores, axis=1).reshape((N, -1))
        probs = np.exp(scores)
        probs /= np.sum(probs, axis=1, keepdims=True)
        loss = -np.sum(np.log(probs[np.arange(N), y]))/N
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2)) 

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # Backward pass, computing the derivatives of the weights                   #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        df = probs.copy()
        df[np.arange(N), y] -= 1.0
        grads['b2'] = (np.sum(df, axis=0)/N)
        grads['W2'] = np.transpose(ReLU_H).dot(df)/N + reg * W2

        dReLU = df.dot(np.transpose(W2))
        dReLU[H < 0] = 0
        grads['b1'] = (np.sum(dReLU, axis=0)/N)
        grads['W1'] = np.transpose(X).dot(dReLU)/N + reg * W1

        return loss, grads

    def train_one_epoch(self, X, y, X_val, y_val, learning_rate=1e-3, reg=1e-5, batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
            X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
            after each epoch.
        - reg: Scalar giving regularization strength.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = int(num_train // batch_size) + 1

        # Use SGD to optimize the parameters in self.model
        loss_history = []

        for it in range(iterations_per_epoch):
            #########################################################################
            # Create a random minibatch of training data and labels, storing        #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            sample_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[sample_indices]
            y_batch = y[sample_indices]
            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
            #########################################################################
            # Use the gradients in the grads dictionary to update the               #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['b2'] -= learning_rate * grads['b2']

            if verbose and it % 25 == 0:
                print('iteration %d / %d: loss %f' % (it, iterations_per_epoch, loss))
 
        val_acc = (self.predict(X_val) == y_val).mean()    

        return {
            'loss_history': loss_history,
            'val_acc': val_acc,
        }

    def predict(self, X):
        """
        Predict labels for data points. For each data point we predict scores for 
        each of the C classes, and assign each data point to the class with the 
        highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
        classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
        the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
        to have class c, where 0 <= c < C.
        """
        H = X.dot(self.params['W1']) + self.params['b1'][None, :]
        ReLU_H = H.copy()
        ReLU_H[H < 0] = 0
        scores = ReLU_H.dot(self.params['W2']) + self.params['b2'][None, :]
        y_pred = np.argmax(scores, axis=1)
        return y_pred
