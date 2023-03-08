import numpy as np
import math
from scipy.optimize import minimize

class periodic_regression:

    def __init__(self,rho):
        self.K = rho.size
        self.rho = rho
        self.tol=1e-6
        self.theta_hat=[]
        self.method="L-BFGS-B"
        
    def f(self,theta,X):
        if len(theta)==2*self.K+1:
            b=theta[-1]
            w=theta[:self.K]
            phi=theta[self.K:2*self.K]
            fx=[]
            for x in X:
                val=0
                for k in range(self.K):
                    val+=w[k]*math.cos((2*math.pi*x/self.rho[k])-phi[k])
                fx.append(b+val)
            return fx

    def risk(self,theta,X,Y):
        Ypred=np.array(self.f(theta,X))
        loss=(np.array(Y)-Ypred)**2
        return np.sum(loss)/len(X)

    def riskGrad(self,theta,X,Y):   
        Y_pred=np.array(self.f(theta,X))
        riskgrad=np.zeros(2*self.K+1)
        for i in range(len(X)):
            loss=Y_pred[i]-Y[i]
            for k in range(self.K):
                riskgrad[k]+=loss*math.cos(2*math.pi*X[i]/self.rho[k]-theta[k+self.K])
            for k in range(self.K,2*self.K):
                riskgrad[k]+=loss*theta[k-self.K]*math.sin(2*math.pi*X[i]/self.rho[k-self.K]-theta[k])
            riskgrad[-1]+=loss
        return riskgrad*2/len(X)

    def fit(self,X,Y):
        x0=np.zeros(2*self.K+1)
        minimizing=minimize(self.risk,x0,args=(X,Y),tol=self.tol,method=self.method,options={"disp":1})
        self.theta_hat=minimizing.x
        return self.theta_hat 