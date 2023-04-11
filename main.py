'''
from colorama import *

def iris_ml_model(sepal_length,sepal_width,petal_length,petal_width):
    """
    #In this ML Model we will detect iris flower
    """

    try:
        # ------------------Importing------------------
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import load_iris

        # ------------------Features and labels------------------
        iris = load_iris()
        features = iris.data
        labels = iris.target

        # ------------------Making Classifier------------------
        clf = DecisionTreeClassifier(criterion='gini',max_depth=3)

        # ------------------Training Classifier------------------
        clf.fit(features,labels)

        # ------------------Test our model------------------
        a = clf.predict([[sepal_length,sepal_width,petal_length,petal_width]])

        flower = ""
        if a==[0]:
            flower ='Setosa'

        elif a==[1]:
            flower ='Versicolor'

        elif a==[2]:
            flower ='Virginica'

        return flower
    except Exception as e:
        print("500! Internal Server Error")

try:
    sepal_length = input("Enter sepal length: ")
    sepal_width = input("Enter sepal width: ")
    petal_length = input("Enter petal length: ")
    petal_width = input("Enter petal width: ")
    print(iris_ml_model(sepal_length,sepal_width,petal_length,petal_width))

except KeyboardInterrupt:
    print(Fore.MAGENTA,"\n\n--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n")
    print(Fore.RED,"500! Internal Server Error")
    print(Fore.GREEN,"\n\n---------------------------------------------------------------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!------------------------------------------------------------------------------------------")
    print(Fore.RESET)
'''





"""
from dotenv import load_dotenv

load_dotenv(dotenv_path="Test.env",override=False,verbose=False,encoding='utf-8')
"""







# Deep Learning

import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2)
model.evaluate(x_test, y_test)