from numpy import array, concatenate, ones, zeros, append, multiply, tanh, identity, var
from numpy.random import seed as random_seed, uniform, shuffle
from numpy.linalg import inv
from scipy.sparse.linalg import eigs
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from colorama import Fore

#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================
#  ESN

# positional arguments
# @ data = time series data to learn
# key word arguments:
# @ rho = rhoscale, spectral width used to scale reservoir matrix W
# @ alpha = leakage rate
# @ beta = regularization used when computing readout matrix Wout
# @ in_nodes = number of input dimensions
# @ out_nodes = number of output dimensions
# @ Ttrain = number of time steps in training
# @ Twashout = number of time steps in reservoir acclimation
# @ N = reservoir size i.e. number of nodes
# @ W_sparsity = proportion of zero-weighted connections in reservoir matrix; 0 = no zeros; 1 = all zeros
# @ Win_height
# @ W_height
# @ seed = random number seed
# @ fix_eig = True to fix consistent W eigen value on reinitialisation; False to allow slightly different values on reinitialisation

class Esn:
    
    def __init__(self, data, **kwargs):
        params = {key:(kwargs[key] if key in kwargs else getattr(Globs, key)) for key in list(vars(Globs))[1:-3]}
        random_seed(params['seed'])            # set the random seed
        # variables
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.out_nodes = params['out_nodes']
        self.N = params['N']
        self.Ttrain = params['Ttrain']
        self.Twashout = params['Twashout']
        self.T1 = self.Twashout                                                     # the first time step after washout
        self.T2 = self.Ttrain + self.Twashout                                        # the first time step after training
        self.data = reshape_data(data)                                              # reshape the data into a row vector
        self.data_input = concatenate(( ones((1,len(data))), self.data ))           # include row of ones - data_input has shape [2,len(data)] with the top row all 1s
        self.inN1 = params['in_nodes'] + self.N + 1
        # matrices
        self.x = zeros(self.N)                                                      # column vector
        self.Win = uniform(low=-params['Win_height'], high=params['Win_height'], size=(self.N,params['in_nodes']+1))
        W = uniform(low=-params['W_height'], high=params['W_height'], size=(self.N,self.N))
        self.W = sparsify(params['W_sparsity'],W,params['seed'])                    # sparsify reservoir
        self.Wout = zeros((self.out_nodes,self.inN1))
        self.M_train = zeros((self.inN1,self.Ttrain))                               # measurement matrix
        self.rescale_W(params['rho'],params['fix_eig'])                             # rescale spectral width of W
        return
        
    # Rescale W given a new rhoscale
    def rescale_W(self, r, fix_eig):
        eig,_ = eigs(self.W,k=1,which='LM',tol=1e-8,v0=ones(self.N)) if fix_eig else eigs(self.W,k=1,which='LM',tol=1e-8)
        self.W = self.W*(r/abs(eig))
        return
    
    #======================================================================================================================================================================
    # CALCULATIONS
    
    # reservoir update calculation
    def update(self, input_data):
        return (1-self.alpha)*self.x + self.alpha*tanh(self.Win@input_data + self.W@self.x)
    
    # prediction of the value at the next time step
    def predict(self, input_data):
        return self.Wout@concatenate((input_data,self.x))
        
    #======================================================================================================================================================================
    # TRAINING
    
    # Composite train function
    def train(self):
        self.train_M()
        self.train_readouts()
        return
        
    # Contruct M matrix
    def train_M(self):
        # store input and expected output in vectors
        self.X_train = self.data[:,self.T1:self.T2]                # input data for the training period
        self.Y_train = self.data[:,self.T1+1:self.T2+1]            # correct output for each corresponding input i.e. the data value at the following time step
        # Washout - sync internal revervoir states with input
        for i in range(self.Twashout):
            self.x = self.update(self.data_input[:,i])             # perform the reservoir update
        # train and extract M matrix
        for i in range(self.Ttrain):
            j = i+self.T1
            self.x = self.update(self.data_input[:,j])             # perform the reservoir update
            self.M_train[:,i] = concatenate((self.data_input[:,j],self.x.flatten()))
        # store the reservoir activtions states after training
        self.x_store = self.x.copy()
        return
    
    # Train Wout (provided M trained)
    def train_readouts(self):
        Mt = self.M_train.transpose()
        D = self.Y_train
        self.Wout = D@Mt@inv(self.M_train@Mt + self.beta*identity(self.inN1))   # output weights solution equation
        self.Yhat_train = self.Wout@self.M_train                                # perform output prediction on established data
        self.nmse_train = nmse(self.Yhat_train,D)                               # compute the error on the predictions w.r.t. the correct data
        return
        
    #======================================================================================================================================================================
    # VALIDATION
    
    # Validate prediction with forced system for T > Ttrain
    def validate(self, val_time=1000):
        # store input and expected output in vectors
        self.X_val = self.data[:,self.T2:self.T2+val_time]               # input data for the validation period, starting at the point after training
        self.Y_val = self.data[:,self.T2+1:self.T2+val_time+1]           # correct output for each corresponding input i.e. the data value at the following time step
        self.Yhat_val = zeros((self.out_nodes, val_time))                # initialise the prediction matrix
        self.x = self.x_store.copy()                                     # return reservoir to freshly trained state
        # test the network on unseen data immediately following the training data, using true data as input
        for i in range(val_time):
            j = i+self.T2
            self.x = self.update(self.data_input[:,j])        # update the reservoir
            y = self.predict(self.data_input[:,j])            # compute the prediction of the value at the next time step
            self.Yhat_val[:,i] = y                            # store the prediction
        self.nmse_val = nmse(self.Yhat_val,self.Y_val)        # compute the error on the predictions w.r.t. the correct data
        return

    #======================================================================================================================================================================
    # TESTING
    
    # Test function given a starting data_point and test time
    def test(self,test_time=1000):
        # store input and expected output in vectors
        self.X_test = self.data[:,self.T2:self.T2+test_time]             # input data for the test period, starting at the point after training
        self.Y_test = self.data[:,self.T2+1:self.T2+test_time+1]         # correct output for each corresponding input i.e. the data value at the following time step
        self.Yhat_test = zeros((self.out_nodes, test_time))              # initialise the prediction matrix
        self.x = self.x_store.copy()                                     # return reservoir to freshly trained state
        y = self.data[:,self.T2]                                         # initialise the prediction value, used as input data for reservoir update, as the first true value in the test data
        # test the network on unseen data immediately following the training data, using predictions as input
        for i in range(test_time):
            u = concatenate([[1.],y])                            # construct input vector from the prediction at the previous time step
            self.x = self.update(u)                              # perform the reservoir update
            y = self.predict(u)                                  # compute the prediction of the value at the next time step
            self.Yhat_test[:,i] = y                              # store the prediction
        self.nmse_test = nmse(self.Yhat_test,self.Y_test)        # compute the error on the predictions w.r.t. the correct data
        return
    
    #======================================================================================================================================================================
    # PLOTTING (copied)

    # Plot the static properties of the ESN
    def plot_static_properties(self):
        fig,_=plt.subplots(nrows=1, ncols=2)
        fig.tight_layout()
        plt.subplot(1,2,1)
        plt.plot(self.Win.transpose())
        plt.title('Win')
        
        plt.subplot(1,2,2)
        plt.pcolor(self.W)
        plt.title('W')
        
    #---------------------------------------------------------------------------------------------
        
    # Plot M matrix
    def plot_M(self):
        fig,_=plt.subplots(nrows=1, ncols=1)
        plt.subplot(3,1,1)
        plt.pcolor(self.M,cmap='RdBu')
        plt.title('M')
        
    #---------------------------------------------------------------------------------------------
    
    # Plot the training data
    def plot_training(self):
        fig,_=plt.subplots(nrows=2*self.out_nodes, ncols=1)
        fig.tight_layout()
        plt.subplot(2*self.out_nodes,1,1)
        plt.title('Training')    
        for i in range(self.out_nodes):
            plt.subplot(2*self.out_nodes,1,1+i*self.out_nodes)
            plt.plot(self.Yhat_train[i,:].T,label='prediction training dataset')
            plt.plot(self.Y_train[i,:].T,label='training dataset')
            plt.legend()

            plt.subplot(2*self.out_nodes,1,2+i*self.out_nodes)
            yy=(self.Yhat_train[i,:]-self.Y_train[i,:])**2
            plt.plot(yy.T,label='NMSE training')
            plt.legend()
            
    #---------------------------------------------------------------------------------------------
            
    # Plot the validation data
    def plot_validation(self):
        fig,_=plt.subplots(nrows=2*self.out_nodes, ncols=1)
        fig.tight_layout()
        plt.subplot(2*self.out_nodes,1,1)
        plt.title('Validation')
            
        for i in range(self.out_nodes):
            plt.subplot(2*self.out_nodes,1,1+i*self.out_nodes)
            plt.plot(self.Yhat_val[i,:].T,label='prediction validation dataset')
            plt.plot(self.Y_val[i,:].T,label='validation dataset')
            plt.legend()

            plt.subplot(2*self.out_nodes,1,2+i*self.out_nodes)
            yy=(self.Yhat_val[i,:]-self.Y_val[i,:])**2
            plt.plot(yy.T,label='NMSE validation')
            plt.legend()
            
    #---------------------------------------------------------------------------------------------
    
    # Plot the test data
    def plot_test(self):        
        fig,_=plt.subplots(nrows=2*self.out_nodes, ncols=1)
        fig.tight_layout()
        plt.subplot(2*self.out_nodes,1,1)
        plt.title('Test')
        for i in range(self.out_nodes):
            plt.subplot(2*self.out_nodes,1,1+i*self.out_nodes)
            plt.plot(self.Yhat_test[i,:].T,label='prediction test dataset')
            plt.plot(self.Y_test[i,:].T,label='test dataset')
            plt.legend()

            plt.subplot(2*self.out_nodes,1,2+i*self.out_nodes)
            yy=(self.Yhat_test[i,:]-self.Y_test[i,:])**2
            plt.plot(yy.T,label='NMSE test')
            plt.legend()

#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================
# OPTIMISER

class Optimiser:
    
    def __init__(self, data, val_time=500, test_time=500):
        self.esn = OptEsn(data)
        self.val_time = val_time
        self.test_time = test_time
        return
    
    def opt_rho(self, rhos):
        print("OPTIMISING RHO...")
        store_r = self.esn.rho
        rnmse = {}
        p = 0
        for rho in rhos:
            self.esn.rho = rho
            self.esn.train()
            self.esn.test(self.test_time)
            rnmse[rho] = self.esn.nmse_test
            p = progress_tracker(p, rho, list(rhos), 10)
        self.esn.rho = store_r
        optr = min_from_dict(rnmse)
        print(Fore.MAGENTA+"OPTIMAL RHO: {}".format(optr)+Fore.RESET)
        return optr, rnmse
    
    def opt_alpha(self, alphas):
        print("OPTIMISING ALPHA...")
        store_a = self.esn.alpha
        anmse = {}
        p = 0
        for alpha in alphas:
            self.esn.alpha = alpha
            self.esn.train()
            self.esn.test(self.test_time)
            anmse[alpha] = self.esn.nmse_test
            p = progress_tracker(p, alpha, list(alphas), 10)
        self.esn.alpha = store_a
        opta = min_from_dict(anmse)
        print(Fore.MAGENTA+"OPTIMAL ALPHA: {}".format(opta)+Fore.RESET)
        return opta, anmse
    
    def opt_rho_alpha(self, rhos, alphas):
        print("CROSS OPTIMISING RHO AND ALPHA...")
        store_r = self.esn.rho
        store_a = self.esn.alpha
        ranmse = []
        p = 0
        for rho in rhos:
            self.esn.rho = rho
            for alpha in alphas:
                self.esn.alpha = alpha
                self.esn.train()
                self.esn.test(self.test_time)
                ranmse.append(((rho,alpha),self.esn.nmse_test))
            p = progress_tracker(p, rho, list(rhos), 5)
        self.esn.rho = store_r
        self.esn.alpha = store_a
        ranmse = sorted(ranmse, key = lambda x: x[1])
        optr, opta = sorted(ranmse, key = lambda x: x[1])[0][0]
        ranmse.insert(0,(('rho','alpha'),'nmse'))
        print(Fore.MAGENTA+"OPTIMAL RHO-ALPHA PAIR: RHO = {}, ALPHA = {}".format(optr, opta)+Fore.RESET)
        return optr, opta, ranmse
    
    def opt_beta(self, betas):
        print("OPTIMISING BETA...")
        store_b = self.esn.beta
        bnmse = {}
        p = 0
        self.esn.train_M()
        for beta in betas:
            self.esn.beta = beta
            self.esn.train_readouts()
            self.esn.validate(self.val_time)
            bnmse[beta] = self.esn.nmse_test
            p = progress_tracker(p, beta, list(betas), 10)
        self.esn.beta = store_b
        optb = min_from_dict(bnmse)
        print(Fore.MAGENTA+"OPTIMAL BETA: {}".format(optb)+Fore.RESET)
        return optb, bnmse
    
#==========================================================================================================================================================================
    
class OptEsn(Esn):
    
    def __init__(self, data, rho=0, alpha=0, beta=0):
        random_seed(Globs.seed)            # set the random seed
        # variables
        self.rho = rho if rho else Globs.rho
        self.alpha = alpha if alpha else Globs.alpha
        self.beta = beta if beta else Globs.beta
        self.out_nodes = Globs.out_nodes
        self.N = Globs.N
        self.Ttrain = Globs.Ttrain
        self.Twashout = Globs.Twashout
        self.T1 = Globs.Twashout                                                          # the first time step after washout
        self.T2 = self.Ttrain+self.Twashout                                                   # the first time step after training
        self.data = reshape_data(data)                                              # reshape the data into a column vector
        self.data_input = concatenate(( ones((1,len(data))), self.data ))           # include row of ones - data_input has shape [2,len(data)] with the top row all 1s
        self.inN1 = Globs.in_nodes + self.N + 1
        # matrices
        self.x = zeros(self.N)                                                           # column vector
        self.x_init = self.x.copy()
        self.Win = uniform(low=-Globs.Win_height, high=Globs.Win_height, size=(self.N,Globs.in_nodes+1))
        W = uniform(low=-Globs.W_height, high=Globs.W_height, size=(self.N,self.N))
        self.W = sparsify(Globs.W_sparsity,W,Globs.seed)                                   # sparsify reservoir
        self.W_init = self.W.copy()
        self.Wout = zeros((self.out_nodes,self.inN1))
        self.M_train = zeros((self.inN1,self.Ttrain))                                    # measurement matrix
        return
    
    # Rescale W given a new rhoscale
    def rescale_W(self):
        eig,_ = eigs(self.W_init,k=1,which='LM',tol=1e-8,v0=ones(self.N)) if Globs.fix_eig else eigs(self.W_init,k=1,which='LM',tol=1e-8)
        self.W = self.W_init*(self.rho/abs(eig))
        return
    
    def reset_x(self):
        self.x = self.x_init.copy()
        return
    
    def train_M(self):
        self.rescale_W()
        self.reset_x()
        super().train_M()
        return

#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================
# GLOBAL VARIABLES

class Globs:

    rho = 1
    alpha = 0.70
    beta = 1e-7
    in_nodes = 1
    out_nodes = 1
    Ttrain = 1000
    Twashout = 200
    N = 100
    W_sparsity = 0.5
    Win_height = 0.5
    W_height = 1.5
    seed = 0
    fix_eig = True
    
#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================
# Other functions

# reshapes a normal iterable into a column vector
def reshape_data(data):
    return array([data])

# nmse function for error calculation
def nmse(y_pred,y_true):
    return mean_squared_error(y_pred,y_true)/var(y_true)

# Reduce a data array to a sparse version (entries = 0) proportional to 'sparsity'
# sparsity = 0 means no change; sparsity = 1 means all zeros
def sparsify(sparsity, data, seed=0):
    assert 0 <= sparsity <= 1, "sparsity must be between 0 and 1"
    random_seed(seed)
    L = data.size                                  # store total length of original data 
    array1 = ones(int(L*(1-sparsity)))             # generate array of ones
    array0 = zeros(int(L*sparsity))                # generate array of zeros
    sparse = concatenate([array1,array0])          # concatenate ones and zeros into sparsity array
    shuffle(sparse)                                # randomize order of ones and zeros
    if sparse.size<L:                              # sometimes one more entry is needed due to loss when rounding using int()
        sparse = append(sparse,1.)
    assert sparse.size == L, "wrong size sparsity array: {}, should be {}".format(sparse.size, L)
    sparse = sparse.reshape(data.shape)            # reshape 1D sparsity array into the shape of the original data array
    return multiply(data,sparse)                   # elementwise mulitiplication - entries in original array are eliminated where sparsity array holds zero

# extract key with lowest value from dictionary d
def min_from_dict(d):
    return list(d.keys())[list(d.values()).index(min(d.values()))]

# used to print a continuous progress tracker as percentages for long running functions
# 'interval' is the minimum % change required for a progress update
# requires percentage = 0 initialised in the outer function
def progress_tracker(percentage, item, sequence, interval=5):
    done = 100*(sequence.index(item) + 1)/len(sequence)                                      # compute how far along 'item' is in 'sequence' as a percentage
    if done-interval >= percentage:                                                    # if 'item' is over one 'interval' % further along than the currently stored percentage:
        new_percentage = int(done-(done%interval))                                       # update current % to reflect the new progress, rounding to the nearest 'interval' %
        print(Fore.CYAN+"    PROCESSING... {}%".format(new_percentage)+Fore.RESET)       # print the progress update message
        return new_percentage                                                            # return the new percentage to be stored in the outer function for the next call to progress tracker
    return percentage                                                                  # if the % progress has not surpassed the next interval, return the percentage unchanged and print nothing

# old init
# def __init__1(self, data, rho=1.25, alpha=0.5, beta=10**-4, in_nodes=1, out_nodes=1, Ttrain=1000, Twashout=100, N=100, W_sparsity=0.5, Win_height=0.5, W_height=1.5, seed=0, fix_eig=True):
#         random_seed(seed)            # set the random seed
#         # variables
#         self.alpha = alpha
#         self.beta = beta
#         self.out_nodes = out_nodes
#         self.N = N
#         self.Ttrain = Ttrain
#         self.Twashout = Twashout
#         self.T1 = Twashout                                                          # the first time step after washout
#         self.T2 = Ttrain+Twashout                                                   # the first time step after training
#         self.data = reshape_data(data)                                              # reshape the data into a row vector
#         self.data_input = concatenate(( ones((1,len(data))), self.data ))           # include row of ones - data_input has shape [2,len(data)] with the top row all 1s
#         self.inN1 = in_nodes + N + 1
#         # matrices
#         self.x = zeros(N)                                                           # column vector
#         self.Win = uniform(low=-Win_height, high=Win_height, size=(N,in_nodes+1))
#         W = uniform(low=-W_height, high=W_height, size=(N,N))
#         self.W = sparsify(W_sparsity,W,seed)                                   # sparsify reservoir
#         self.Wout = zeros((out_nodes,self.inN1))
#         self.M_train = zeros((self.inN1,Ttrain))                                    # measurement matrix
#         self.rescale_W(rho,fix_eig)                                                 # rescale spectral width of W
#         return