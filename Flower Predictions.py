#Importing modules
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle

#Pulling in csv file
dataFolder = r'C:\Eshan Study\VS Code\Flower Predictions\adaptediris.csv' #Reads in the CSV file, must use r'' as the file path contains backslashes

#Reading the file data
df = pd.read_csv(dataFolder) #Reads the file

#Summarize the data

#Shape
print(df.shape) #Prints the number of rows and columns in the dataset

#Prints (150, 5) indicating there are 150 columns and 5 rows

#Prints a summary of the dataset
print(df.head(20)) #Prints the headers of the data and the first 20 data rows

#Statistics of the data
print(df.describe()) #This gives us some statistics on the data

#Class distribution
print(df.groupby('variety').size()) #This tells us how many instances there are of each variety

#Data visualization

#Univariate plots

#Box-and-whiskers plot

df.plot(kind = 'box', subplots = True, layout = (2, 2), sharex = False, sharey = False) #Creates the box-and-whiskers plot

#Kind indicates what type of graph is needed
#Subplots groups columns in the data into it's own plot
#Layout of the subplots visually
#Sharex sets x axis label
#Sharey sets y axis label

pyplot.show() #Prints the plots

#Histogram

df.hist()
pyplot.show() #Prints the plot

#Using the graphs we can see the data has a normal distribution

#Multivariate plots

#Scatter plot matrix

scatter_matrix(df) #Creates a scatter plot matrix
pyplot.show() #Prints it

#The matrix pits every attribute against each other to find a correlation

#Creating our testing and training datasets

array = df.values #Create an array from all the values in the dataset

dataInput = array[:,0:4] #Takes columns 0(sepal length), 1(sepal width), 2 (petal length), and 3 (petal width)

#dataInput is a variable that stores an array of all the input data

dataOutput = array[:,4] #Takes column 4 (variety)

#dataOutput is a variable that stores and arry of all the output data

#We are now splitting our data into testing and training data

#We split the data with 80% training and 20% testing
#We include a random state to ensure the data is randomized and the split remains constant

dataInputTrain, dataInputTest, dataOutputTrain, dataOutputTest = train_test_split(dataInput, dataOutput, test_size = 0.2, random_state = 50)

#Sample data entries from the training sets
print(dataInputTrain[0:5])
print(dataOutputTrain[0:5])

#This is what will train the model

#Sample data entries from the testing set
print(dataInputTest[0:5])
print(dataOutputTest[0:5])

#This is the data that tests the accuracy of the model

#Spot Check Algorithms

#We save the name of a model and call the machine learning model into a list by combining them into one item in the list

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = [] #Results of calling the models
names = [] #Name of the model

for (name, model) in models: #Iterates through the items in the models list

    #Shuffles the data and divides it into 10 groups

    kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)

    #Takes in the model, divides the training data into the 10 groups, trains the model on the 9 groups
    #It then tests the models accuracy on the tenth group
    
    cv_results = cross_val_score(model, dataInputTrain, dataOutputTrain, cv = kfold, scoring = 'accuracy')

    #Saves the resulting accuracy to a list
    
    results.append(cv_results)

    #Saves the name of the model tested to a list
    
    names.append(name)

    #Prints the name of the model, the average accuracy, and the standard deviation of the accuracy
    
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#Evaluate the models' accuracy

#Compare algorithms

pyplot.boxplot(results, labels = names)
pyplot.title('Comparison of Algorithms')
pyplot.show()

#Make predictions

#Create final model
model = LinearDiscriminantAnalysis() #Choose your algorithm
model.fit(dataInputTrain, dataOutputTrain) #Train the model on the training sets
predictions = model.predict(dataInputTest) #Input the testing set

#Evauldate predictions
print(accuracy_score(dataOutputTest, predictions)) #Shows the accuracy: 100%
print(confusion_matrix(dataOutputTest, predictions)) #Shows the number of false positives and negatives
print(classification_report(dataOutputTest, predictions)) #Shows various scores for the model

#Make predictions

#Create final model
model = SVC(gamma = 'auto') #Choose your algorithm
model.fit(dataInputTrain, dataOutputTrain) #Train the model on the training sets
predictions = model.predict(dataInputTest) #Input the testing set

#Evauldate predictions
print(accuracy_score(dataOutputTest, predictions)) #Shows the accuracy: 100%
print(confusion_matrix(dataOutputTest, predictions)) #Shows the number of false positives and negatives
print(classification_report(dataOutputTest, predictions)) #Shows various scores for the model

#Save your model
with open('flowerPredictionModel.pickle', 'wb') as flowerPredictionModel:

    pickle.dump(model, flowerPredictionModel)
