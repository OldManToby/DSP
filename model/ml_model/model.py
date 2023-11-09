#Toby Warn (20026345)

#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

def load_csv():
    data = pd.read_csv(r'C:\Users\Toby\Documents\GitHub\DSP\team_stats\2022\Offensive\stats_1.csv')
    return data

data = load_csv()

print (data)
