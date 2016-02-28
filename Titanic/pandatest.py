# ====== basic package
import csv as csv
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import pylab as P

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# machine learning package
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ======= import test and train dataset
titanic_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)


# ======= drop uncessary cols just drop Embark col for it seems no relatinoship to the survival rate
titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_df = test_df.drop(['Name', 'Ticket'], axis=1)

titanic_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)


# ======= fare value
# filling missing value in test dataset of fare
test_df.Fare.fillna(test_df.Fare.median(), inplace=True)

# convert fare into int
titanic_df.Fare = titanic_df.Fare.astype(int)
test_df.Fare = test_df.Fare.astype(int)

# get fare for survived & didn't survive passengers 
fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]
fare_survived     = titanic_df["Fare"][titanic_df["Survived"] == 1]

# get average and std for fare of survived/not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])

# plot
titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))

avgerage_fare.index.names = std_fare.index.names = ["Survived"]
avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)

# ======= Age 
# using mean and std to generate random age value to fill in missing value
average_age_titanic = titanic_df.Age.mean()
std_age_titanic = titanic_df.Age.std()
count_nan_age_titanic = titanic_df.Age.isnull().sum()

average_age_test = test_df.Age.mean()
std_age_test = test_df.Age.std()
count_nan_age_test = test_df.Age.isnull().sum()

# generate between mean-std & mean+std
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# fill in random value into missing Age
titanic_df.Age[np.isnan(titanic_df.Age)] = rand_1
test_df.Age[np.isnan(test_df.Age)] = rand_2

# convert to int
titanic_df.Age = titanic_df.Age.astype(int)
test_df.Age = test_df.Age.astype(int)

# ======== Cabin: has too many missing so just drop it
titanic_df.drop(['Cabin'], axis=1,inplace=True)
test_df.drop(['Cabin'], axis=1,inplace=True)

# ======== Family: combine sibsp and parch and if has family set it to 1 and vice versa
titanic_df['Family'] = titanic_df.Parch + titanic_df.SibSp
titanic_df.Family.loc[titanic_df.Family > 0] = 1
titanic_df.Family.loc[titanic_df.Family == 0] = 0

test_df['Family'] = test_df.Parch + test_df.SibSp
test_df.Family.loc[test_df.Family > 0] = 1
test_df.Family.loc[test_df.Family == 0] = 0

# drop parch and sibsp
titanic_df.drop(['Parch', 'SibSp'], axis=1)
test_df.drop(['Parch', 'SibSp'], axis=1)

# ======== Sex: add a new value called children when age < 16 --> more significant
def get_person(passenger):
	age,sex = passenger
	return 'child' if age < 16 else sex

# create a new col called person and apply the function to set its value to child or sex
titanic_df['Person'] = titanic_df[['Age', 'Sex']].apply(get_person, axis=1)
test_df['Person'] = test_df[['Age', 'Sex']].apply(get_person, axis=1)

# and drop sex col
titanic_df.drop(['Sex'], axis=1, inplace=True)
test_df.drop(['Sex'], axis=1, inplace=True)

# create dummy var for Person and drop Male --- low average
person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])
person_dummies_titanic.columns = ['Male','Female','Child']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Male','Female','Child']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

titanic_df = titanic_df.join(person_dummies_titanic)
test_df    = test_df.join(person_dummies_test)


titanic_df.drop(['Person'],axis=1,inplace=True)
test_df.drop(['Person'],axis=1,inplace=True)

# =========== Pclass
# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

titanic_df.drop(['Pclass'],axis=1,inplace=True)
test_df.drop(['Pclass'],axis=1,inplace=True)

titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df    = test_df.join(pclass_dummies_test)

# =========== define training and testing sets
# separate survived to training the data
X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()

# =========== Logistic Regression -- better!
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

print logreg.score(X_train, Y_train)

# =========== Random Forest
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

print random_forest.score(X_train, Y_train)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)