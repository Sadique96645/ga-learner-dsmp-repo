# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data = pd.read_csv(path)
data['Gender'].replace('-','Agender',inplace=True)
gender_count = data['Gender'].value_counts()
plt.bar(gender_count.index,gender_count)
plt.show()
#Code starts here 




# --------------
#Code starts here
alignment = data['Alignment'].value_counts()
plt.pie(alignment,labels = alignment.index)
plt.title('Character Alignment')



# --------------
#Code starts here
sc_df = data[['Strength','Combat']].copy()
sc_covariance = sc_df.Strength.cov(sc_df.Combat)
sc_strength = sc_df.Strength.std()
sc_combat = sc_df.Combat.std()
sc_pearson = sc_covariance/(sc_strength*sc_combat)
print(sc_pearson)

ic_df = data[['Intelligence','Combat']].copy()
ic_covariance = ic_df.Intelligence.cov(ic_df.Combat)
ic_intelligence = ic_df.Intelligence.std()
ic_combat = ic_df.Combat.std()
ic_pearson = ic_covariance/(ic_intelligence*ic_combat)
print(ic_pearson)


# --------------
#Code starts here
total_high = data.Total.quantile(q=0.99)

print(total_high)
super_best = data[(data.Total > total_high)]
print(super_best)
super_best_names =super_best.Name.values.tolist()
print(super_best_names)


# --------------
#Code starts here
fig,(ax_1,ax_2,ax_3) = plt.subplots(3,1,figsize= (20,10))
ax_1.boxplot(super_best['Intelligence'])
ax_1.set(title= 'Intelligence')
ax_2.boxplot(super_best['Speed'])
ax_2.set(title= 'Speed')
ax_3.boxplot(super_best['Power'])
ax_3.set(title= 'Power')
plt.show()


