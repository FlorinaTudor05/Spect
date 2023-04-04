import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import svm
#citire fisier test
df= pd.read_csv('SPECT.train', header=None,
                    names=['OVERALL_DIAGNOSIS','F1', 'F2',  'F3', 'F4','F5', 'F6', 'F7','F8','F9','F10','F11','F12','F13','F14', 'F15','F16','F17','F18', 'F19', 'F20', 'F21', 'F22']
)
train=df.to_numpy()#conversie dataframe la array
y=train[:,1]#slice y, pastrez prima coloana din dataframe
X=train[0:,1:]#slice x, pastrez restul coloanelor
#Fac citire de baza de date si split in X,y si pentru fisierul de test
df1= pd.read_csv('SPECT.test', header=None,
                    names=['OVERALL_DIAGNOSIS','F1', 'F2',  'F3', 'F4','F5', 'F6', 'F7','F8','F9','F10','F11','F12','F13','F14', 'F15','F16','F17','F18', 'F19', 'F20', 'F21', 'F22']
)
test=df1.to_numpy()
y_test=test[:,1]
X_test=test[0:,1:]
model=svm.SVC(kernel='linear', C=1.0).fit(X, y)#initializare SVM
y_pred = model.predict(X_test)
print("C=1")
print("predictie:")
print(y_pred)
print("acuratete:")
print(accuracy_score(y_test, y_pred))
#variez costul
print('Variere cost')
for i in  [-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]:
    model1=svm.SVC(kernel='linear', C=pow(2,i)).fit(X, y)
    print("C=")
    print( pow(2,i))
    y_pred = model1.predict(X_test)
    print("predictie:")
    print(y_pred)
    print("acuratete:")
    print( accuracy_score(y_test, y_pred))
