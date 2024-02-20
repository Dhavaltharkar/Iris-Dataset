import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle  

import warnings; warnings.filterwarnings('ignore')

# Load the Dataset
data = pd.read_csv("https://raw.githubusercontent.com/Dhavaltharkar/TSF_Data_Science_and_Business_Analytics/main/TSF_GRIP_Prediction_Using_Unsupervised_ML/Iris.csv")

# Splitting the dataset into independent variable X and dependent variable y
X = data.iloc[:, [0, 1, 2, 3]]
y = data.Species

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Making a Support Vector Machine Model
svclassifier = SVC(kernel='poly', degree=8, gamma='auto')
svclassifier.fit(X_train, y_train)

# Save the Decision tree model as a pickle file
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svclassifier, f)
