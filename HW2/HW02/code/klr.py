import numpy as np
import scipy.special 
from scipy.optimize import minimize

from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)

class klr:

    def __init__(self,Xtr,lam=0.1,beta=0.1):
        self.lam  = lam
        self.beta = beta
        self.Xtr  = Xtr
        self.tol  = 1e-6
        self.method = "L-BFGS-B"
        self.theta_hat = []
        
    def compute_kernel(self,X,Xprime):
        #X.shape[0] is column and Xprime.shape[0] is rows
        krbf=np.array([[np.exp(-self.beta*np.linalg.norm(X[i]-Xprime[j])**2) for i in range(X.shape[0])] for j in range(Xprime.shape[0])])
        return krbf
    
    def discriminant(self,theta,X):
        K=self.compute_kernel(self.Xtr,X)
        return theta[-1]+K@np.array(theta[:-1])

    def risk(self,theta,X,Y):
        self.discriminant=memory.cache(self.discriminant)
        self.compute_kernel=memory.cache(self.compute_kernel)
        Discriminant=self.discriminant(theta,X)
        K=self.compute_kernel(X,X)

        theta=np.array(theta)
        n=len(X)
        r1=np.sum([np.log(1+np.exp(-Y[i]*Discriminant[i])) for i in range(n)])
        r2=np.sum([theta[i]*theta[j]*K[i][j] for i in range(n) for j in range(n)])*self.lam/2
        return (r1+r2)/n

    def risk_grad(self,theta,X,Y,K=0):
        gradient=[]
        n=len(X)
        self.discriminant=memory.cache(self.discriminant)
        self.compute_kernel=memory.cache(self.compute_kernel)
        Discriminant=self.discriminant(theta,X)
        K=self.compute_kernel(X,X)
        grad_b=[]
        for i in range(n):
            inner_gradient=[]
            for j in range(n):
                inner_gradient.append((1/(np.exp(Y[i]*Discriminant[i])+1)*-Y[i]*K[i][j]
                                +self.lam*theta[j]*K[i][j])/n)
            gradient.append(sum(inner_gradient))
            grad_b.append(-Y[i]/(1+np.exp(Y[i]*Discriminant[i])))
        gradient.append((np.sum(grad_b))/n)
        return gradient

    def fit(self,X,Y):
        theta0=np.zeros(len(X)+1)
        minimizing=minimize(self.risk,theta0,args=(X,Y),tol=self.tol,method=self.method,options={"disp":1,"maxfun":1e+10,"maxiter":1000})
        self.theta_hat=minimizing.x
        return minimizing.fun
        
        
    def predict(self,theta,X):
        return np.sign(self.discriminant(theta,X))
    
    def classification_error(self,theta,X,Y):
        ce=0
        Y_pred=self.predict(theta,X)
        for i in range(len(Y)):
            if Y[i]==Y_pred[i]:
                ce+=1
        return (1-ce/len(Y))*100        