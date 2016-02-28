# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB

# get crime dataset 
crime_df = pd.read_csv('./seattle_incidents_summer_2014.csv', low_memory=False)

# get a sense of dataset
crime_df.head()

# drop necessary cols for this analysis
crime_df = crime_df.drop(['General Offense Number', 'RMS CDW ID', 'Offense Code', 'Offense Code Extension', 'Summary Offense Code', 'Date Reported', 'Occurred Date Range End'], axis=1)

crime_df['Summarized Offense Description'].value_counts()

crime_df['Month'].value_counts().plot(kind='line')

crime_df.head()

crime_df['District/Sector'].value_counts().plot(kind='bar')

crime_df['Summarized Offense Description'].value_counts().plot(kind='bar')

crime_df.loc[crime_df['Summarized Offense Description'].isin(['VEHICLE THEFT','BIKE THEFT','ROBBERY','MAIL THEFT'])]

crime_df.loc[crime_df['Summarized Offense Description'].isin(['VEHICLE THEFT','BIKE THEFT','ROBBERY','MAIL THEFT'])]['District/Sector'].value_counts().plot(kind='bar')

crime_df['Occurred Date or Date Range Start'] = pd.to_datetime(crime_df['Occurred Date or Date Range Start'])

crime_df['Occurred Date or Date Range Start'].describe()

crime_df = crime_df.set_index('Occurred Date or Date Range Start')

crime_df.head()

crime_df.between_time('1900', '0600')['Summarized Offense Description'].value_counts().plot(kind='bar')

crime_df['time'] = crime_df.index.hour
crime_df['time']

crime_df.loc[crime_df['Summarized Offense Description'].isin(['ROBBERY'])]['time'].value_counts().sort_index().plot(kind='bar')