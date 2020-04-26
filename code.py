# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# code starts here
df = pd.read_csv(path)
# check the null values
df.isnull().sum()
# drop null values
df.dropna(subset=['Description','CustomerID'],inplace=True)
# check the null values
df.isnull().sum()
# only take one country
df = df[df.Country== 'United Kingdom']

# create new colums returns
df['Return']=df.InvoiceNo.str.contains('C')
# store the result in purchase 
df['Purchase'] = np.where(df["Return"]==True,0,1)
# code ends here


# --------------
# code starts here
# create new dataframe customer
customers = pd.DataFrame({'CustomerID': df['CustomerID'].unique()},dtype=int)
# calculate the recency
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Recency'] = pd.to_datetime("2011-12-10") - (df['InvoiceDate'])
# remove the time factor
df.Recency = df.Recency.dt.days
# purchase equal to one 
temp = df[df['Purchase']==1]
# customers latest purchase day
recency=temp.groupby(by='CustomerID',as_index=False).min()
customers=customers.merge(recency[['CustomerID','Recency']],on='CustomerID')



# code ends here


# --------------
# code stars here
temp_1 = df[['CustomerID','InvoiceNo','Purchase']]
temp_1.drop_duplicates(subset = ['InvoiceNo'] ,inplace = True)
annual_invoice = temp_1.groupby(by = 'CustomerID',as_index= False).sum()
annual_invoice.rename(columns={"Purchase": "Frequency"})
customers = customers.merge(annual_invoice, on='CustomerID')
# code ends here


# --------------
# code starts here
df['Amount'] = df['Quantity'] * df['UnitPrice']
annual_sales = df.groupby(by = 'CustomerID',as_index= False).sum()
annual_sales.rename(columns={"Amount":"monetary"})
annual_sales.head()
customers = customers.merge(annual_sales[['CustomerID','Amount']], on='CustomerID')
# code ends here


# --------------
# code ends here
customers['monetary'] = np.where(customers['monetary']<0,0,customers['monetary'])
customers['Recency_log'] = np.log(customers['Recency']+0.1)
customers['Frequency_log'] = np.log(customers['Frequency']+0.1)
customers['Monetary_log'] = np.log(customers['monetary']+0.1)
# code ends here


# --------------
# import packages
from sklearn.cluster import KMeans
# code starts here
dist = []
for i in range(1,10):
    km = KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(customers.iloc[:,1:7])
    dist.append(km.inertia_)
# Initialize figure
# Line plot # clusters on X-axis and WCSS on Y-axis 
plt.plot(range(1,10),dist)
plt.title('The elbow method')
plt.xlabel('The number of clusters')
plt.ylabel('WCSS')
plt.show()


# code ends here


# --------------
# code starts here
from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10,random_state=0)
customers['cluster'] = cluster.fit_predict(customers.iloc[:,1:7])
customers.plot.scatter(x= 'Frequency_log',y='Monetary_log',c='cluster', colormap='viridis')
# code ends here


