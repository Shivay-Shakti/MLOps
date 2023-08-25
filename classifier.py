import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

df = pd.read_csv("/content/MLOps/data/Iris.csv")

features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

target = 'Species'

X_train, x_test, y_train, y_test = train_test_split(df[features],df[target],test_size=0.3,shuffle=True)

# step -1: initialize the model class

clf = DecisionTreeClassifier(criterion='entropy') #information gain as criterion

# step -2: train the model on training set 

clf.fit(X_train,y_train)

# step -3: evaluate the data on testing set 

y_pred = clf.predict(x_test)

print(f"Accuracy of the model is {accuracy_score(y_test,y_pred)*100}") # -> this tests accuracy







