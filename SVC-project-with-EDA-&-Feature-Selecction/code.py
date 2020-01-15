# --------------
import pandas as pd
from collections import Counter

# Load dataset
data = pd.read_csv(path)
data.isnull().sum()


# --------------
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style(style='darkgrid')

# Store the label values 
label = data.iloc[:,-1]
label.head(5)
sns.countplot(data=data,x='Activity')

# plot the countplot



# --------------
import numpy as np
# make the copy of dataset
data_copy = data.copy()
mask = ('WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS')
# Create an empty column 
data_copy['duration'] = ''
# Calculate the duration
duration_df = data_copy.groupby([label.mask(label!= 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS'), 'subject'])['duration'].count() * 1.28
duration_df = pd.DataFrame(duration_df)
plot_data = duration_df.sort_values(by='duration',ascending= False)
plot_data.reset_index(inplace=True)
replaced_value = {'WALKING_UPSTAIRS':'Upstairs','WALKING_DOWNSTAIRS':'Downstairs'}
plot_data['Activity'] = plot_data['Activity'].map(replaced_value)
sns.barplot(data=plot_data,x='subject',y='duration')
# Sort the values of duration





# --------------
#exclude the Activity column and the subject column
feature_cols = data.select_dtypes(exclude=['object','int']).columns
#Calculate the correlation values
correlated_values = data[feature_cols].corr().stack().reset_index()
#stack the data and convert to a dataframe
correlated_values = pd.DataFrame(correlated_values)
correlated_values.rename(columns = {'level_0':'Feature_1','level_1':'Feature_2',0:'Correlation_score'},inplace=True)
#create an abs_correlation column
correlated_values['abs_correlation'] = correlated_values['Correlation_score'].abs()
#Picking most correlated features without having self correlated pairs
s_corr_list = correlated_values.sort_values(by='abs_correlation',ascending=False)
top_corr_fields = s_corr_list[(s_corr_list['abs_correlation']>0.8)]
top_corr_fields = top_corr_fields[(top_corr_fields['Feature_1'])!=(top_corr_fields['Feature_2'])]
print(top_corr_fields.head())


# --------------
# importing neccessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as error_metric
from sklearn.metrics import confusion_matrix, accuracy_score

# Encoding the target variable
le = LabelEncoder()
le.fit(data['Activity'])
data['Activity'] = le.transform(data['Activity'])
# split the dataset into train and test
X = data.drop('Activity',1)
y = data['Activity']
X_train, X_test, y_train ,y_test = train_test_split(X,y,test_size=0.3,random_state=40)
classifier = SVC()
clf = classifier.fit(X_train,y_train)
y_pred = clf.predict(X_test)
precision, accuracy , f_score, _ = error_metric(y_test,y_pred,average = 'weighted')
model1_score = accuracy_score(y_test,y_pred)
print(precision)
print(accuracy)
print(f_score)
print(model1_score) 




# --------------
# importing libraries
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
# Feature selection using Linear SVC
lsvc = LinearSVC(C=0.01,penalty = 'l1',dual = False,random_state =42)
lsvc.fit(X_train,y_train)
model_2 = SelectFromModel(lsvc,prefit=True)
new_train_features= model_2.transform(X_train)
new_test_features = model_2.transform(X_test)
classifier_2 = SVC()
clf_2 = classifier_2.fit(new_train_features,y_train)
y_pred_new = clf_2.predict(new_test_features)
model2_score = accuracy_score(y_test,y_pred_new)
precision, accuracy , f_score, _ = error_metric(y_test,y_pred_new,average = 'weighted')
# model building on reduced set of features




# --------------
# Importing Libraries
from sklearn.model_selection import GridSearchCV

# Set the hyperparmeters
parameters = {'kernel':['linear','rbf'],'C':[100, 20, 1, 0.1]}


# Usage of grid search to select the best hyperparmeters
svc = SVC()
selector = GridSearchCV(svc,parameters,scoring='accuracy')
selector.fit(new_train_features,y_train)
print(selector.best_params_)
print(selector.cv_results_)
means = selector.cv_results_['mean_test_score']
stds = selector.cv_results_['std_test_score']
params = selector.cv_results_['params']
print(means,stds,params)

classifier_3 = SVC(C=20,kernel='rbf')
clf_3 = classifier_3.fit(new_train_features,y_train)
y_pred_final = clf_3.predict(new_test_features)
model3_score = accuracy_score(y_test,y_pred_final)
precision,recall,f_score,_ = error_metric(y_test,y_pred_final,average='weighted')
print(precision)
print(recall)
print(f_score)
print(model3_score)
    

# Model building after Hyperparameter tuning





