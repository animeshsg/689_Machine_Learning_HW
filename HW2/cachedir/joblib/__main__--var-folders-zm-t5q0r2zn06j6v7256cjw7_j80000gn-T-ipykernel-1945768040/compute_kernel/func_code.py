# first line: 1
    def compute_kernel(self,X,Xprime):
        #X.shape[0] is column and Xprime.shape[0] is rows
        krbf=np.array([[np.exp(-self.beta*np.linalg.norm(X[i]-Xprime[j])**2) for i in range(X.shape[0])] for j in range(Xprime.shape[0])])
        return krbf
