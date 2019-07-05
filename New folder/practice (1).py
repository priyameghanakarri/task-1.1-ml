# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:31:09 2019

@author: HP
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import datetime,calendar
import numpy as np
def convert(time):
     dt = datetime.datetime.strptime(time, "%Y%m%d-%H:%M")
     t=calendar.timegm(dt.utctimetuple())
     return t
def scale(x): 
    x1=[i[0] for i in x]
    n=float(len(x))
    avg=sum(x1)/n
    x=[[(i-avg)/n] for i in x1]
    return x,avg
def rms(a,b):
    err=0.0
    for i in range(0,len(a)):
        err=err+np.square(a[i]-b[i])
    m=float(len(a))
    err=err/m
   
    return np.sqrt(err)
data=pd.read_csv('testset.csv')
############ temperature  ################
x=data.iloc[0:100000,0:1].values
y=data.iloc[0:100000,11:12].values
x_test=data.iloc[75000:78000,0].values
y_test=data.iloc[75000:78000,11:12].values

y=[str(i[0]) for i in y]

xt=[]
yt=[]
for i in range(0,len(x)):    
    if y[i]!='nan':
            yt.append([float(y[i])])
            xt.append([x[i][0]])

x=xt
y=yt
x=[[convert(i[0])] for i in x]

print (x)

model=LinearRegression()

from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=30)
x_new=poly_reg.fit_transform(x)
model.fit(x_new,y)

plt.scatter(x,y)
plt.plot(x,model.predict(x_new),'r')
plt.show()

inp="20190815-10:00"
inp=convert(inp)
print (inp)
print (model.predict(poly_reg.fit_transform([[inp]])))
#y_p=model.predict(x_test)
#print rms(y_p,y_test)

###########  pressure ####################
x=data.iloc[0:100000,0:1].values
y=data.iloc[0:100000,8:9].values
x_test=data.iloc[75000:78000,0].values
y_test=data.iloc[75000:78000,8:9].values

y=[str(i[0]) for i in y]

xt=[]
yt=[]
for i in range(0,len(x)):    
    if y[i]!='nan':
            yt.append([float(y[i])])
            xt.append([x[i][0]])

x=xt
y=yt
x=[[convert(i[0])] for i in x]

print (x)

model=LinearRegression()

from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=30)
x_new=poly_reg.fit_transform(x)
model.fit(x_new,y)

plt.scatter(x,y)
plt.plot(x,model.predict(x_new),'r')
plt.show()

inp="20160815-10:00"
inp=convert(inp)
print (inp)
print (model.predict(poly_reg.fit_transform([[inp]])))
#y_p=model.predict(x_test)
#print rms(y_p,y_test)
############### humidity ###################
x=data.iloc[0:100000,0:1].values
y=data.iloc[0:100000,6:7].values
x_test=data.iloc[75000:78000,0].values
y_test=data.iloc[75000:78000,6:7].values

y=[str(i[0]) for i in y]

xt=[]
yt=[]
for i in range(0,len(x)):    
    if y[i]!='nan':
            yt.append([float(y[i])])
            xt.append([x[i][0]])

x=xt
y=yt
x=[[convert(i[0])] for i in x]

print (x)

model=LinearRegression()

from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=30)
x_new=poly_reg.fit_transform(x)
model.fit(x_new,y)

plt.scatter(x,y)
plt.plot(x,model.predict(x_new),'r')
plt.show()

inp="20190815-10:00"
inp=convert(inp)
print (inp)
print (model.predict(poly_reg.fit_transform([[inp]])))
#y_p=model.predict(x_test)
#print rms(y_p,y_test)

############## wind speed #####################
x=data.iloc[0:100000,0:1].values
y=data.iloc[0:100000,19:20].values
x_test=data.iloc[75000:78000,0].values
y_test=data.iloc[75000:78000,19:20].values

y=[str(i[0]) for i in y]

xt=[]
yt=[]
for i in range(0,len(x)):    
    if y[i]!='nan':
            yt.append([float(y[i])])
            xt.append([x[i][0]])

x=xt
y=yt
x=[[convert(i[0])] for i in x]

print (x)

model=LinearRegression()

from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=30)
x_new=poly_reg.fit_transform(x)
model.fit(x_new,y)

plt.scatter(x,y)
plt.plot(x,model.predict(x_new),'r')
plt.show()

inp="20170815-10:00"
inp=convert(inp)
print (inp)
print (model.predict(poly_reg.fit_transform([[inp]])))
#y_p=model.predict(x_test)
#print rms(y_p,y_test)

############### visibility ##################
x=data.iloc[0:100000,0:1].values
y=data.iloc[0:100000,14:15].values
x_test=data.iloc[75000:78000,0].values
y_test=data.iloc[75000:78000,14:15].values

y=[str(i[0]) for i in y]

xt=[]
yt=[]
for i in range(0,len(x)):    
    if y[i]!='nan':
            yt.append([float(y[i])])
            xt.append([x[i][0]])

x=xt
y=yt
x=[[convert(i[0])] for i in x]

print (x)

model=LinearRegression()

from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=30)
x_new=poly_reg.fit_transform(x)
model.fit(x_new,y)

plt.scatter(x,y)
plt.plot(x,model.predict(x_new),'r')
plt.show()

inp="20170815-10:00"
inp=convert(inp)
print (inp)
print (model.predict(poly_reg.fit_transform([[inp]])))
#y_p=model.predict(x_test)
#print rms(y_p,y_test)

################# dew point #########################
x=data.iloc[0:100000,0:1].values
y=data.iloc[0:100000,2:3].values
x_test=data.iloc[75000:78000,0].values
y_test=data.iloc[75000:78000,2:3].values

y=[str(i[0]) for i in y]

xt=[]
yt=[]
for i in range(0,len(x)):    
    if y[i]!='nan':
            yt.append([float(y[i])])
            xt.append([x[i][0]])

x=xt
y=yt
x=[[convert(i[0])] for i in x]

print (x)

model=LinearRegression()

from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=30)
x_new=poly_reg.fit_transform(x)
model.fit(x_new,y)

plt.scatter(x,y)
plt.plot(x,model.predict(x_new),'r')
plt.show()

inp="20170815-10:00"
inp=convert(inp)
print (inp)
print (model.predict(poly_reg.fit_transform([[inp]])))
#y_p=model.predict(x_test)
#print rms(y_p,y_test)

########################## heat index ####################
x=data.iloc[0:100000,0:1].values
y=data.iloc[0:100000,5:6].values
x_test=data.iloc[75000:78000,0].values
y_test=data.iloc[75000:78000,5:6].values

y=[str(i[0]) for i in y]

xt=[]
yt=[]
for i in range(0,len(x)):    
    if y[i]!='nan':
            yt.append([float(y[i])])
            xt.append([x[i][0]])

x=xt
y=yt
x=[[convert(i[0])] for i in x]

print (x)

model=LinearRegression()

from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=30)
x_new=poly_reg.fit_transform(x)
model.fit(x_new,y)

plt.scatter(x,y)
plt.plot(x,model.predict(x_new),'r')
plt.show()

inp="20170815-10:00"
inp=convert(inp)
print (inp)
print (model.predict(poly_reg.fit_transform([[inp]])))
#y_p=model.predict(x_test)
#print rms(y_p,y_test)



                    # Logistic Regression ##
def scale1(x): 
    x1=[i[0] for i in x]
    n=float(len(x))
    avg=sum(x1)/n
    x=[[(i-avg)/n] for i in x1]
    return x

data=pd.read_csv('testset.csv')
x=data.iloc[0:60000,0].values
x_test=data.iloc[60000:100000,0].values
y1=data.iloc[0:60000,9].values
y1_test=data.iloc[60000:100000,9].values

x=[[convert(i)] for i in x]
x_test=[[convert(i)] for i in x_test]

x=scale1(x)
x=scale1(x)
x_test=scale1(x_test)
x_test=scale1(x_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x,y1)

plt.scatter(x,y1)
plt.plot(x,model.predict(x),'r')
plt.show()
y_pred=model.predict(x_test)
print (rms(y_pred,y1_test))

# Logistic Regression 2 ##
data=pd.read_csv('testset.csv')
x=data.iloc[0:60000,0].values
x_test=data.iloc[60000:100000,0].values
y1=data.iloc[0:60000,3].values
y1_test=data.iloc[60000:100000,3].values

x=[[convert(i)] for i in x]
x_test=[[convert(i)] for i in x_test]

x=scale(x)
x=scale(x)
x_test=scale(x_test)
x_test=scale(x_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x,y1)

plt.scatter(x,y1)
plt.plot(x,model.predict(x),'r')
plt.show()
y_pred=model.predict(x_test)
print (rms(y_pred,y1_test))



 # Logistic Regression 3 ##
data=pd.read_csv('testset.csv')
x=data.iloc[0:60000,0].values
x_test=data.iloc[60000:100000,0].values
y1=data.iloc[0:60000,12].values
y1_test=data.iloc[60000:100000,12].values

x=[[convert(i)] for i in x]
x_test=[[convert(i)] for i in x_test]

x=scale(x)
x=scale(x)
x_test=scale(x_test)
x_test=scale(x_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x,y1)

plt.scatter(x,y1)
plt.plot(x,model.predict(x),'r')
plt.show()
y_pred=model.predict(x_test)
print (rms(y_pred,y1_test))



                    ## Multi Classification ##

x2=data.iloc[0:60000,[2,3,6,8,9,11,12,14,19]].values
y2=data.iloc[0:60000,1:2].values

x2=[[str(i[0]),str(i[1]),str(i[2]),str(i[3]),str(i[4]),str(i[5]),str(i[6]),str(i[7]),str(i[8])] for i in x2]
#print x2
xt=[]
yt=[]

for i in range(0,len(x2)): 
    
    if x2[i][0]!='nan' and x2[i][2]!='nan' and x2[i][2]!='-9999' and x2[i][3]!='N/A'and x2[i][4]!=''and x2[i][5]!=''and x2[i][6]!='':
            if x2[i][7]!='nan'and x2[i][8]!='nan' :
                yt.append([y2[i][0]])
                xt.append([float(x2[i][0]),float(x2[i][1]),float(x2[i][2]),float(x2[i][3]),float(x2[i][4]),float(x2[i][5]),float(x2[i][6]),float(x2[i][7]),float(x2[i][8])])
                
                #print 'i=',i
    
print (xt)
x2=xt
y2=yt

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaler.fit(x2)

x2=scaler.transform(x2)
from sklearn.neural_network import MLPClassifier

mlp=MLPClassifier(hidden_layer_sizes=(9,9,9),max_iter=5000)
mlp.fit(x2,y2)

print (mlp.predict([[14.13,980.24,54.16,9.76,3.70,28.234,0,2.5,7.4]]))






