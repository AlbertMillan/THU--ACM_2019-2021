import numpy as np

def softmax_classifier(W, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N, 1), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    # TODO: Put your code here
#     print("INPUT:",input.shape)
#     print("W:",W.shape)                 
#     print("label:",label.shape)
    
    # Pre-processing
    N, D = input.shape
    C = W.shape[1]
    
    # Scores
    s = np.dot(input,W)                 # (NxC)
    prediction = s.argmax(axis=1)       # (N,)
    
    # Loss
    e_s = np.exp(s)                     # (NxC)
    e_y = np.sum( (e_s * label), axis=1 )   # (N,)
    e_sum = np.sum(e_s, axis=1)         # (N,)
    
    loss = np.sum( -np.log( e_y / e_sum ) )
    
    # Gradient
    ds = e_s / e_sum[:,None]            # (NxC)
    ds -= label                         # (NxC)
    gradient = np.dot(input.T, ds)
    
    # Average loss
    loss /= N
    gradient /= N

    # Regularization
    loss += lamda * np.sum(W*W)
    gradient += lamda * 2 * W

    ############################################################################

    return loss, gradient, prediction
