# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
p_a = len(df[df['fico']>700])/len(df)
print(p_a)
p_b = len(df[df['purpose']=='debt_consolidation'])/len(df)
df1 = len(df[df['purpose']=='debt_consolidation'])
p_a_b = (p_a * p_b)/p_a
p_b_a = (p_a * p_b)/p_b
result = p_a_b == p_a
print(result)
# code ends here


# --------------
# code starts here
prob_lp = df[df['paid.back.loan'] == 'Yes'].shape[0]/df.shape[0]
prob_cs = df[df['credit.policy'] == 'Yes'].shape[0]/df.shape[0]
new_df = df[df['paid.back.loan']== 'Yes']
prob_pd_cs = new_df[new_df['credit.policy'] == 'Yes'].shape[0] / new_df.shape[0]
bayes = (prob_pd_cs * prob_lp)/prob_cs
print(bayes)

# code ends here


# --------------
# code starts here
df1 = df[df['paid.back.loan'] == 'No']
df1.purpose.value_counts(normalize=True).plot(kind='bar')



# code ends here


# --------------
# code starts here
inst_median = df['installment'].median()
print(inst_median)
inst_mean = df['installment'].mean()
print(inst_mean)
plt.hist([inst_median, inst_mean])


# code ends here


