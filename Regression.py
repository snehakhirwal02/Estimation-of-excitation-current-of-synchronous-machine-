# Import libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
# import statsmodels.api as sm
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
import streamlit as st #streamlit backend

# Class variables: Y_test, Y_pred,reg, X_train,Y_train

# Methods:
#          model() -- Splits the data into train and test dataset, fits the training data to the model and uses RandomForestRegressor to get the prediction for 
#                     testing data using the best parameter values obtained by performing grid search optimisation 
#          result() -- To get the Root mean squared error(RMSE) and R-squared score for the model
#          prediction_plot() -- To display the 2D-plot for the actual vs predicted values
#          gridSearchCV()  -- For hyperparameter tuning 

# User inputs:
#          estimator -- a paramter in the RandomForestRegressor which denotes the number of trees (Range : 10 to 1000 )
#          test_size -- proportion of the original dataset to be included in the test split  (Range: 0% to 100 % )

# Outputs:
#          Root mean squared error(RMSE)
#          R-squared score
#          2-D plot : Optimal prediction vs real prediction


class Regressor:
    """Class Regressor: Random forest Regressor
    This Class contains the methods for Random forest Regressor 
    Class input parameters:
    :param df: The input data frame
    :type df: Pandas DataFrame
    :param estimator: Number of decision trees to be build before taking the average of all predictions to be specified by the user
    :type estimator: Integer
    :param test_size: User Input - Proportion of test data specified by user in which dataset is to be splitted.
    :type test_size: float
    Class Output Parameters:
    :param Y_pred: The resulting output of the Regression test
    :type Y_pred: float 
    :param Y_test: The expected output of the Regression test
    :type Y_test: float  
    :param R squared score: Model accuracy on the Training data
    :type: float 
    :param RMSE: Root mean squared error   
    :type RMSE: float 
    :param Error_message: Error message if an exception was encountered during the processing of the code
    :type Error_message: str 
    :param flag: internal flag for marking if an error occurred while processing a previous method
    :type flag: bool 
    """

    Y_test=None
    Y_pred=None
    reg=None
    X_train=None
    Y_train=None
    Error_message=None
    flag=None

    # instance attribute
    def __init__(self,X,Y):
        """Class Constructor
        :param X: Features to train the model
        :type X: array
        :param Y: Target variable that is to be classified
        :type Y: array
        
        """
        self.X = X
        self.Y = Y

    def model(self,estimator, test_size):
        """ model Method : 
            This method splits the data into train and test sets, then creates a model based on the user input n_estimator and test_size. 
            
            It calls model 'gridSearchCV' that returns the best parameters on which the model can be fitted.
            
            It then fits the model based on the best parameters obtained after Grid search cross validation and test it on the test dataset, then returns the predicted value 'Y_pred' 
           
            :param estimator: User Input - Number of decision trees to be build before taking the average of all prediction.
            :type estimator: Integer
            :param test_size: User Input - Proportion of test data specified by user in which dataset is to be splitted.
            :type test_size: float
        """
        try:
            self.estimator=estimator 
            self.test_size=test_size  
            # Split the data into training set and testing set
            X_train,X_test,Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=test_size,random_state=123)      
            
            # Create a model
            reg=RandomForestRegressor(n_estimators=estimator)

            # Fitting training data to the model
            reg.fit(X_train,Y_train)

            self.Y_pred = reg.predict(X_test)
            b=[]
            b=self.gridSearchCV(reg, X_train, Y_train)
            # re-create the model
            reg=RandomForestRegressor(n_estimators=estimator,max_depth=b[0],min_samples_split=b[1],min_samples_leaf=b[2],max_features=b[3])

            # Fitting training data to the model
            reg.fit(X_train,Y_train)
            self.Y_pred = reg.predict(X_test)
        except Exception as e:
            self.Error_message = 'Error while creating model: ' +str(e)
            self.flag=True
            st.warning(self.Error_message)
            self.Y_pred=[]

        return(self.Y_pred)

    def result(self,Y_test, Y_pred): 
        """ model result : 
            This method displays metrics 'R-squared score' and 'RMSE - Root Mean Squared Error' value to analyze the performace of model,
            
            R-Squared Score: It is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. 
            RMSE: Mean Squared Error represents the average of the squared difference between the original and predicted values in the data set. 
            It measures the variance of the residuals. Root Mean Squared Error is the square root of Mean Squared error. It measures the standard deviation of residuals.
 
            :param Y_test: The resulting output of the Regression test
            :type Y_test: float
            :param Y_pred: The expected output of the Regression test
            :type Y_pred: float
        """ 
        # To compute R-squared score+
        if self.flag != True:
            try:
                r2 = r2_score(Y_test, Y_pred)
                st.metric('R-squared score: ', round(r2, 4))
                # To compute root mean squared error
                st.metric('RMSE: ', round((MSE(Y_test,Y_pred)**(0.5)), 4))
                # To compute adjusted R-squared error 
                # r_adj = 1 - ((1-r2)*((Y_test.shape[0])-1))/(Y_test.shape[0]-X_test.shape[1]-1)
                # print('R-squared adjusted:',r_adj)
            except Exception as e:
                self.Error_message = 'Error while printing outputs: ' +str(e)
                self.flag=True
                st.warning(self.Error_message)
        else:
            st.write('Error occurred in previous methods, Refer to Error Message Warning')

            
    def prediction_plot(self,Y_test, Y_pred):
        """ prediction_plot Method : 
                This method prints a plot that represents the real prediction and optimal prediction. 
                Optimal prediction are the true values that are plotted as a line plot and real prediction are values predicted by the classification model that are plotted as a scatter plot.
        """
        if self.flag != True:
            try:
                # To display the 2D-plot for the actual vs predicted values
                df=pd.DataFrame({'y_test':Y_test,'y_pred':Y_pred})
                fig = plt.figure(figsize=(10, 4))
                sns.scatterplot(x='y_test',y='y_pred',data=df,label='Real Prediction')
                sns.lineplot(x='y_test',y='y_test',data=df,color='red',alpha=0.5,label='Optimal Prediction')
                plt.title('y_test vs y_pred')
                plt.legend()
                st.pyplot(fig)
            except Exception as e:
                self.Error_message = 'Error while plotting: ' +str(e)
                self.flag=True
                st.warning(self.Error_message)
        else:
            st.write('Error occurred in previous methods, Refer to Error Message Warning')

    def gridSearchCV(self,reg,X_train, y_train):
        """ gridSearchCV Method : 
                This method returns the best parameters on which the model is to be fitted.
                Searched Parameters:  
                            1. max_features:      Number of features to consider at every split
                            2. max_depth :        Maximum number of levels in tree
                            3. min_samples_split: Minimum number of samples required to split a node
                            4. in_samples_leaf:   Minimum number of samples required at each leaf node
                :return: Best parameter values on which regressor is fitted.
        """
        # Find the best parameters for the model using Grid search optimisation
        parameters = {
       'max_depth': [70, 80, 90, 100],
       'min_samples_split':[2,5,10],
       'min_samples_leaf':[1,2,4],
       'max_features': ["auto", "sqrt", "log2"],
        }
        gridforest = GridSearchCV(reg, parameters, cv = 3, n_jobs = -1, verbose = 1)
        gridforest.fit(X_train, y_train)
        best = gridforest.best_params_
        #Storing the best parameter values to pass as paramters in the function 'model'
        a=best['max_depth']
        b=best['min_samples_split']
        c=best['min_samples_leaf']
        d=best['max_features']
        return(a,b,c,d)
       

