
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load hospital_billing = hb
hb = pd.read_csv('C:/Users/vince_000/Documents/GitHub/Sim_V_MzW/Data/Hospital_Billing_time_and_event_filtered.csv')

hb.columns

# Adjust date format
hb['completeTime'] = hb['completeTime'].astype("datetime64") 


#ix_complete_time = list(hb.columns).index('completeTime')
#ix_case = list(hb.columns).index('case')

# Get Information about the dataset

## Timespan

hb.filter(['case']).groupby(hb['completeTime'].dt.year).count().plot(kind = "bar") 


## Filter on the most current cases year 2015 and 2016)

hb = hb[(hb['completeTime'].dt.year == 2015) |  (hb['completeTime'].dt.year == 2016)]

# Analysis: Count of events per user

event_user = hb.filter(['case','event','org:resource'])
event_user_pivot = event_user.pivot_table(values = 'case', index = 'org:resource', columns = 'event',  aggfunc='count', fill_value = 0)

#events_per_user = event_user.pivot_table(index = 'org:resource', aggfunc = 'count', fill_value = 0)
#plt.hist(events_per_user['event'], bins = 100, density = True)


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,50):
    kmeans = KMeans(n_clusters = i,
                    init = 'k-means++',
                    max_iter = 300,
                    n_init = 10,
                    random_state = 0)
    kmeans.fit(event_user_pivot)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,50),wcss)
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

    
    
plt.plot(range(5,50),wcss[4:51])
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means to the hospital dataset
kmeans = KMeans(n_clusters = 25,
                init = 'k-means++',
                max_iter = 300,
                n_init = 10,
                random_state = 0)
y_kmeans = kmeans.fit_predict(event_user_pivot)

# Adding the cluster to the users as groups
c_event_user_pivot = event_user_pivot.filter(['org:resource'])
c_event_user_pivot['cluster'] = y_kmeans

hb = hb.merge(right = c_event_user_pivot, how = 'left', left_on = 'org:resource' , right_index = True)

cluster_user = hb.filter(['case','event','cluster'])
cluster_user_pivot = event_user.pivot_table(values = 'case', index = 'cluster', columns = 'event',  aggfunc='count', fill_value = 0)
