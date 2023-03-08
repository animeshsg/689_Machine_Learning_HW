import numpy as np
import matplotlib.pyplot as plt
from regression import periodic_regression

#Load the data
data = np.load("../data/data.npz")
Xtr = data["Xtr"]
Ytr = data["Ytr"]
Xte = data["Xte"]
Yte = data["Yte"]
Ytr=Ytr.reshape(Ytr.shape[0])
Xtr=Xtr.reshape(Xtr.shape[0])
Yte=Yte.reshape(Yte.shape[0])
Xte=Xte.reshape(Xte.shape[0])

#Define the component periods
rho=np.array([12.42, 12.00, 12.66, 23.93, 25.82, 6.21, 4.14, 6.00, 6.10])

#Instantiate the model
model = periodic_regression(rho)

#Fit the model
theta_hat=model.fit(Xtr,Ytr)
theta0=np.ones(2*model.K+1)


##Risk for Training Data
risk=model.risk(theta0,Xtr,Ytr)
print(risk)

#Training and Test loss
aslosstr=model.risk(theta_hat,Xtr,Ytr)
aslosste=model.risk(theta_hat,Xte,Yte)
print("Training loss:{}  Test loss:{}".format(aslosstr,aslosste))

# first 48 Hours of Training data - Predicted vs Actual 
Xe=Xtr[:48]
Ye=Ytr[:48]
Ye_pred=model.f(theta_hat,Xe)
plt.scatter(Xe,Ye,color = 'red')
plt.plot(Xe,Ye_pred)
plt.show()

# Last 48 hours of test data - predicted vs actual
Xe=Xte[-48:]
Ye=Yte[-48:]
Ye_pred=model.f(theta_hat,Xe)
plt.scatter(Xe,Ye,color = 'red')
plt.plot(Xe,Ye_pred)
plt.show()

