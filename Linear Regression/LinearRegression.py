import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib import animation # this is the first time i tried this lib from matplotlib

# below URL which is having the x and y values
url='https://raw.githubusercontent.com/AshishJangra27/Machine-Learning-with-Python-GFG/main/Linear%20Regression/data_for_lr.csv'

#dependent variable or target variable
#independent variable or num of features
df=pd.read_csv(url)
#print(df.isnull().sum())

# Drop the missing values
df=df.dropna() #if you wan to fill empty values or null values you can just out fillna(df.mean()) 

#traning dataset and labels
train_input=np.array(df.x[0:500]).reshape(500,1)
train_output=np.array(df.y[0:500]).reshape(500,1)

#valid dataset with labels
test_input=np.array(df.x[500:700]).reshape(199,1)
test_ouput=np.array(df.y[500:700]).reshape(199,1)

print(train_input)

#now we have a splited training dataset and testing data values
#now we are gonna ready to build Linear Regression model

class LinearRegression:
    def __init__(self) :
        self.parameters={}
    
    def forward_propagation(self,train_input):
        m=self.parameters['m']
        c=self.parameters['c']
        predictions=np.multiply(m,train_input)+c  # simply tell y= mx+c # you already know about this or y=theta1+theta1(x)
        #above like B0+B1(X)=> B0 is intercept B1=slope of the variable
        return predictions
    def cost_function(self,predictions,train_output):
        cost=np.mean((train_output-predictions)**2) #(average of squared error between actual value and predicted value)
        return cost
    def backward_propagation(self,train_input,train_output,predictions):
        derivative={}
        d_f=(train_output-predictions)*-1
        d_m=np.mean(np.multiply(train_input,d_f))
        d_c=np.mean(d_f)
        derivative['dm']=d_m
        derivative['dc']=d_c
        return derivative
    def update_parameters(self,derivative,learning_rate):
        self.parameters['m']=self.parameters['m']-\
            learning_rate*derivative['dm']
        self.parameters['c']=self.parameters['c']-\
            learning_rate*derivative['dc']
    def train(self,train_input,train_output,learning_rate,iters):
        #initialize the random parameters
        self.parameters['m']=np.random.uniform(0,1)*-1
        self.parameters['c']=np.random.uniform(0,1)*-1

        #initialize loss
        self.loss=[]

        #initialize  figure and axis for animation
        fig,ax=plt.subplots()
        x_vals=np.linspace(min(train_input),max(train_input),100)
        line,=ax.plot(x_vals,self.parameters['m']*x_vals+
                      self.parameters['c'],color='red',label='Regression Line')

        ax.scatter(train_input,train_output,marker='o',color='green',label='Training Data')


        #set y-axis limits to exclude - values
        ax.set_ylim(0,max(train_output)+1)

        def update(frame):
            #Forward propagation
            predictions=self.forward_propagation(train_input)

            #costfunction
            cost=self.cost_function(predictions,train_output)

            #back propagation
            derivatives=self.backward_propagation(train_input,train_output,predictions)

            self.update_parameters(derivatives,learning_rate)

            #update the regression line
            line.set_ydata(self.parameters['m']
                           * x_vals+self.parameters['c']) 
            #append loss and print
            self.loss.append(cost)

            print("Iteration={},Loss={}".format(frame+1,cost))

            return line,
        #create animation
        ani=animation.FuncAnimation(fig,update,frames=iters,interval=200,blit=True)

        #save the animation as video file
        writergif = animation.PillowWriter(fps=30)
        ani.save('linear regression.gif',writer=writergif)

        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()

        return self.parameters,self.loss
    

linear_reg=LinearRegression()
parameters,loss=linear_reg.train(train_input,train_output,0.0001,20)
    
    














