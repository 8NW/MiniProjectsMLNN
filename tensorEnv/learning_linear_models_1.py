import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict_label = "G3"

X = np.array(data.drop([predict_label], 1))
Y = np.array(data[predict_label])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

best_model_score = 0

''' # This can be commented out once a solid model is found

for _ in range(50):  # range can differ based on how long you want to check for improved models
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)

        linear_accuracy = linear.score(x_test, y_test)
        print(str(linear_accuracy*100) + "%")  # Shows the accuracy at which the linear model can predict grades

        if linear_accuracy > best_model_score:
                best_model_score = linear_accuracy
                with open("learning_linear_models_1_student_model.pickle", "wb") as f:
                        pickle.dump(linear, f)
                        
'''
pickle_imported = open("learning_linear_models_1_student_model.pickle", "rb")
linear = pickle.load(pickle_imported)

print("Coefficients:\n", linear.coef_)  # The larger the coefficient the larger the specific data type affects the prediction
print("Intercept:\n", linear.intercept_)

predictions = linear.predict(x_test)  # This will take the array of the arrays and test it on data not used for training

for pred_increment in range(len(predictions)):
        print(predictions[pred_increment], x_test[pred_increment], y_test[pred_increment])
        # in the above print, the first value is the predicted, the second is the values used to predict, the thrird is the actual

p = "G1" # This can be changed to find the inputs correlated to the prediction
style.use("ggplot")

pyplot.scatter(data[p], data[predict_label])
pyplot.xlabel(p)
pyplot.ylabel(predict_label)

pyplot.show()






