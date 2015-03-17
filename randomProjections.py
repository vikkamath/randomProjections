#!/share/imagedb/kamathv1/anaconda/bin/python

import numpy as np
import cPickle as pickle
from sklearn import random_projection

#################
## LEGEND:
#################
# params[0] = input -> hidden weights
# params[1] = input -> hidden biases
# params[2] = hidden -> softmax weights
# params[3] = hidden -> softmax biases 
# params[4] = hidden -> hidden weights 
# params[5] = hidden -> softmax biases
# params[6] = shape information of projected hidden-hidden weights
# params[7] = eps that was used to generate projection matrices
#################





def projectMatrices(filename):
    #Load parameters from the 'Large, Trained' model 
    a = pickle.load(open(filename,'rb'))
    #Generate epsilon (tolerance) values from .4 to 1.0
    #             Values less than .4 yield 
    z = [i/100.0 for i in range(40,100)]
    for i in z: 
        params = []
        #Perform transform and check what dimension the resultant will be
        transformer = random_projection.GaussianRandomProjection(eps=i)
        #Find the shape of the matrix that you get with a certain tolerance. 
        #   Shape depends on eps. This shape information is then used
        #   to create a new projection with shape as a parameter (as opposed to eps)
        #TODO: See what happens when some other weight matrix is used to find shape
        a_new = transformer.fit_transform(a[4])
        print 'With eps %f, size of projected weight matrix is %d x %d' % (i,a_new.shape[1],a_new.shape[1])
    
        ##################
        # Project Matrices
        ##################
        #Use shape as a parameter instead of eps - to keep matrices
        #   for weights and biases compatible
        transformer2 = random_projection.GaussianRandomProjection(
                                n_components=a_new.shape[1])
    
        #Project the weight matrix from the input to the first hidden layer 
        params.append(transformer2.fit_transform(a[0]))
        #Project the corresponding bias
        temp = transformer2.fit_transform(np.reshape(a[1],(1,a[1].shape[0])))
        params.append(np.ravel(np.transpose(temp)))
        #Project the weights from the second hidden layer to the softmax layer
        params.append(np.transpose(transformer2.fit_transform(np.transpose(a[2]))))
        params.append(a[3]) #I'm doing all this BS because there was some garbage collection
                            #  issue that I didn't have the patience to work around
        #Project the weight from the first hidden layer to the second hidden layer
        temp = transformer2.fit_transform(a[4])
        #Also project the rows of the matrix
        params.append(np.transpose(transformer2.fit_transform(np.transpose(temp))))
        #Project the biases leading into the second hidden layer
        temp = transformer2.fit_transform(np.reshape(a[5],(1,a[5].shape[0])))
        params.append(np.ravel(np.transpose(temp)))
    
        #NOTE: the biases leading into the softmax don't have to be projected :)
    #    a.append([i])
        
        ############
        #Save Params
        ###########
    
        #Add eps information to the parameters - just for the heck of it
        params.append(np.asarray(a_new.shape))
#        params.append(np.asarray([temp.shape[1],temp.shape[1]]))
        params.append(np.asarray([i]))
        filename="./parameters2/parameters_"+str(a_new.shape[1])+"_"+str(a_new.shape[1])+".pkl"
        pickle.dump(params,open(filename,'wb'))


if __name__=="__main__":
    projectMatrices("parameters.pkl")





