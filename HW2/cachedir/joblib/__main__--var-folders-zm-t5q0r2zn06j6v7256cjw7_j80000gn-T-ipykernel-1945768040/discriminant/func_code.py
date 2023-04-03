# first line: 1
    def discriminant(self,theta,X):
        K=self.compute_kernel(self.Xtr,X)
        return theta[-1]+K@np.array(theta[:-1])
