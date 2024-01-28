import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv("/kaggle/input/kepler-labelled-time-series-data/exoTrain.csv")
test_data = pd.read_csv("/kaggle/input/kepler-labelled-time-series-data/exoTest.csv")

print(train_data.head())
print(train_data.shape)
print(test_data.head())

class_counts = train_data['LABEL'].value_counts()
print(class_counts)
# Check all the missing values
print(train_data.isnull().sum().sum())
train_data['mean'] = train_data.iloc[:, 1:].mean(axis=1)
train_data['std'] = train_data.iloc[:, 1:].std(axis=1)
train_data['skew'] = train_data.iloc[:, 1:].skew(axis=1)
train_data['kurtosis'] = train_data.iloc[:, 1:].kurtosis(axis=1)


train_data[['mean', 'std', 'skew', 'kurtosis']].hist()
plt.show()
train_data[['mean', 'std', 'skew', 'kurtosis']].boxplot(whis=1.5)
plt.show()
# Plot histogram for brightness variations
plt.hist(train_data['mean'], bins=50)
plt.xlabel('Mean Brightness')
plt.ylabel('Frequency')
plt.show()

test_data['mean'] = test_data.iloc[:, 1:].mean(axis=1)
test_data['std'] = test_data.iloc[:, 1:].std(axis=1)
test_data['skew'] = test_data.iloc[:, 1:].skew(axis=1)
test_data['kurtosis'] = test_data.iloc[:, 1:].kurtosis(axis=1)

X_train = train_data.drop(['LABEL'], axis=1)
y_train = train_data['LABEL']
X_test = test_data.drop(['LABEL'], axis=1)
y_test = test_data['LABEL']

print(y_train.dtype)


smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

class_counts_resampled = y_train_resampled.value_counts()
print(class_counts_resampled)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train_resampled, test_size=0.25, random_state=34
)





#Decision Tree
ds_model = DecisionTreeClassifier()

ds_model.fit(X_train_split, y_train_split)

y_val_pred = ds_model.predict(X_val_split)

# Print our evaluation
print(classification_report(y_val_split, y_val_pred))


#Decision Tree
ds_model = DecisionTreeClassifier()

ds_model.fit(X_train_split, y_train_split)

y_val_pred = ds_model.predict(X_val_split)

# Print our evaluation
print(classification_report(y_val_split, y_val_pred))

train_data = pd.read_csv("/input/kepler-labelled-time-series-data/exoTrain.csv")
test_data = pd.read_csv("/input/kepler-labelled-time-series-data/exoTest.csv")

print(train_data.head())
print(train_data.shape)
print(test_data.head())
class_counts = train_data['LABEL'].value_counts()
print(class_counts)
# Check all the missing values
print(train_data.isnull().sum().sum())
train_data['mean'] = train_data.iloc[:, 1:].mean(axis=1)
train_data['std'] = train_data.iloc[:, 1:].std(axis=1)
train_data['skew'] = train_data.iloc[:, 1:].skew(axis=1)
train_data['kurtosis'] = train_data.iloc[:, 1:].kurtosis(axis=1)
train_data[['mean', 'std', 'skew', 'kurtosis']].hist()
plt.show()
train_data[['mean', 'std', 'skew', 'kurtosis']].boxplot(whis=1.5)
plt.show()
# Plot histogram for brightness variations
plt.hist(train_data['mean'], bins=50)
plt.xlabel('Mean Brightness')
plt.ylabel('Frequency')
plt.show()
test_data['mean'] = test_data.iloc[:, 1:].mean(axis=1)
test_data['std'] = test_data.iloc[:, 1:].std(axis=1)
test_data['skew'] = test_data.iloc[:, 1:].skew(axis=1)
test_data['kurtosis'] = test_data.iloc[:, 1:].kurtosis(axis=1)
X_train = train_data.drop(['LABEL'], axis=1)
y_train = train_data['LABEL']
X_test = test_data.drop(['LABEL'], axis=1)
y_test = test_data['LABEL']
print(y_train.dtype)
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)
class_counts_resampled = y_train_resampled.value_counts()
print(class_counts_resampled)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train_resampled, test_size=0.25, random_state=34
)

fig = plt.figure(figsize=(15,40))
plt.title('Stars with Exoplanets')
for i in range(12):
    ax = fig.add_subplot(14,1,i+1)
    train_data[train_data['LABEL']==2].iloc[i,1:].hist(bins=40)


fig = plt.figure(figsize=(15,40))
plt.title('Stars with Exoplanets')
for i in range(12):
    ax = fig.add_subplot(14,1,i+1)
    train_data[train_data['LABEL']==1].iloc[i,1:].hist(bins=40)
    
#Decision Tree
ds_model = DecisionTreeClassifier()

ds_model.fit(X_train_split, y_train_split)

y_val_pred = ds_model.predict(X_val_split)

# Print our evaluation
print(classification_report(y_val_split, y_val_pred))

cm = confusion_matrix(y_val_split, y_val_pred)
print(cm)

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('True Label', fontsize=16)
plt.title('Confusion Matrix', fontsize=24)
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Assuming you have your features (X_train_split, X_val_split) and target labels (y_train_split, y_val_split) already defined

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val_split)

# Initialize the Logistic Regression model with an increased max_iter
# You can adjust the value of max_iter based on your needs
lr_model = LogisticRegression(max_iter=1000)

# Fit the model to your scaled training data
lr_model.fit(X_train_scaled, y_train_split)

# Predict on the scaled validation data
y_val_pred = lr_model.predict(X_val_scaled)

# Print the evaluation report
print(classification_report(y_val_split, y_val_pred))
cm = confusion_matrix(y_val_split, y_val_pred)
print(cm)

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True,cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('True Label', fontsize=16)
plt.title('Confusion Matrix', fontsize=24)
plt.show()
