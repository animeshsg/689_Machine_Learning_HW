import numpy as np
import matplotlib.pyplot as plt
from klr import klr

data = np.load("../data/data.npz")
Xtr = data["Xtr"] #Train inputs
Ytr = data["Ytr"] #Train labels
Xte = data["Xte"] #Test inputs
Yte = data["Yte"] #Test labels

#a
model = klr(Xtr[:3,:],lam=0.1,beta=0.1)
print("K:", model.compute_kernel(Xtr[:3,:],Xte[:4,:]))

#b
model=klr(Xtr[:3])
print(model.discriminant([1,1,1,-0.5],Xte[:4]))

#c
model=klr(Xtr[:3])
print(model.predict([1,1,1,-0.5],Xte[:4]))

#d
model=klr(Xtr[:3])
print(model.risk([1,1,1,-0.5],Xtr[:3],Ytr[:3]))

#e
model=klr(Xtr[:3])
print(model.risk_grad([1,1,1,-0.5],Xtr[:3],Ytr[:3]))

#f
model=klr(Xtr[:10])
test=model.fit(Xtr[:10],Ytr[:10])
print(model.theta_hat)

#g
model=klr(Xtr)
fit=model.fit(Xtr,Ytr)
cetr=model.classification_error(model.theta_hat,Xtr,Ytr)
cete=model.classification_error(model.theta_hat,Xte,Yte)
print(f'training Classification Error:{cetr}    Test Classification error:{cete}')

#i
Xvl=Xtr[:52]
Xtr2=Xtr[53:]
Yvl=Ytr[:52]
Ytr2=Ytr[53:]

values = [0.0001,0.001,0.01,0.1,1,10]
low_err=np.inf
best_beta=values[0]
best_lmbda=values[0]
loss_data=[]
val_ce_data=[]
tr_ce_data=[]
for beta in values:
    loss_vals=[]
    val_ce_vals=[]
    tr_ce_vals=[]
    for lmbda in values:
        model=klr(Xtr2,beta,lmbda)
        fit=model.fit(Xtr2,Ytr2)
        vlce=model.classification_error(model.theta_hat,Xvl,Yvl)
        trce=model.classification_error(model.theta_hat,Xtr2,Ytr2)
        loss_vals.append(fit)
        val_ce_vals.append(vlce)
        tr_ce_vals.append(trce)
        print(f"Beta:{beta} Lambda: {lmbda} Validation Classification error: {vlce} Training classification error: {trce}")
        memory.clear()
        if vlce<low_err:
            low_err=vlce
            best_beta=beta
            best_lmbda=lmbda
    loss_data.append(loss_vals)
    val_ce_data.append(val_ce_vals)
    tr_ce_data.append(tr_ce_vals)

print(f'Best Beta is:{best_beta} Best lambda is: {best_lmbda}')

import matplotlib.pyplot as plt

values = [0.0001,0.001, 0.01, 0.1, 1, 10]

for i in range(len(val_ce_data)):
    plt.plot(values, val_ce_data[i], label=f"beta={values[i]}")

plt.xlabel('Lambda values (log scale)')
plt.ylabel('Classification error')
plt.title('Classification Error(Validation Set) vs. Hyperparameters')
plt.legend()
plt.xscale('log')
plt.show()

import matplotlib.pyplot as plt

values = [0.0001,0.001, 0.01, 0.1, 1, 10]

for i in range(len(tr_ce_data)):
    plt.plot(values, tr_ce_data[i], label=f"beta={values[i]}")

plt.xlabel('Lambda values (log scale)')
plt.ylabel('Classification error')
plt.title('Classification Error (Training set) vs. Hyperparameters')
plt.legend()
plt.xscale('log')
plt.show()

#j
best_model=klr(Xtr,beta=best_beta,lam=best_lmbda)
fit=best_model.fit(Xtr,Ytr)

best_trce=best_model.classification_error(best_model.theta_hat,Xtr,Ytr)
best_tece=best_model.classification_error(best_model.theta_hat,Xte,Yte)
print(f'Training classification error:{best_trce}   Test Classification_error :{best_tece}')
