# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Code starts here
data = pd.read_csv(path)
data.hist(column = 'Rating',bins=8,figsize=(10,10))
data = data[data['Rating'] <=5]
data.hist(column = 'Rating',bins=8,figsize=(10,10))
#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()
percent_null = total_null/data.isnull().count()
missing_data = pd.concat([total_null,percent_null],axis=1,keys=['Total','Percent'])

data = data.dropna(axis=1)
total_null_1 = data.isnull().sum()
percent_null_1 = total_null/data.isnull().count()
missing_data_1 = pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total','Percent'])
print(missing_data_1)
# code ends here


# --------------

#Code starts here
sns.catplot(x="Category",y="Rating",data=data, kind="box" , height = 10)
plt.title('Rating vs Category [BoxPlot]')
plt.xticks(rotation=90)
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
data['Installs'].value_counts()
data['Installs'] = data['Installs'].str.replace('+','')
data['Installs'] = data['Installs'].str.replace(',','')
data['Installs'] = data['Installs'].astype(int)
le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])
sns.regplot(x="Installs", y="Rating", data=data)
plt.title("Rating vs Installs [RegPlot]")
plt.show()
#Code ends here



# --------------
#Code starts here
data['Price'].value_counts()
data['Price'] = data['Price'].str.replace('$','')
data['Price'] = data['Price'].astype(float)
sns.regplot(x="Price", y="Rating", data=data)
plt.title("Rating vs Price [RegPlot]")
plt.show()
#Code ends here


# --------------

#Code starts here
data['Genres'].unique()
data['Genres'] = data['Genres'].str.split(';',n=1,expand = True)[0]
gr_mean = data.groupby(['Genres'],as_index=False)['Genres','Rating'].mean()
print(gr_mean.describe())
gr_mean = gr_mean.sort_values(by=['Rating'])
print(gr_mean.iloc[0])
print(gr_mean.iloc[-1])
#Code ends here


# --------------

#Code starts here
data['Last Updated'].value_counts()
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()
data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days
sns.regplot(x="Last Updated Days", y="Rating", data=data)
plt.title('Rating vs Last Updated [RegPlot]')
#Code ends here


