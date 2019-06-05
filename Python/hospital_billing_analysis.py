
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
cluster_user_pivot = cluster_user.pivot_table(values = 'case', index = 'cluster', columns = 'event',  aggfunc='count', fill_value = 0)

# Calculating throughput times

min_completion_times = pd.read_csv('C:/Users/vince_000/Documents/GitHub/Sim_V_MzW/Data/Created/min_completion_times.txt')
min_completion_times['min_completion_time'] = min_completion_times['min_completion_time'].astype("datetime64") 
# create min_completion_times_per_case (mctpc)
mctpc = min_completion_times.pivot(values= 'min_completion_time', index = '_case_', columns = '_event_')

# now comes the real calculation considering the process flow

mctpc['document_ok'] = ~ pd.isnull(mctpc['DELETE'])
mctpc['diagnosis_correct'] = ~ pd.isnull(mctpc['CHANGE DIAGN'])

mctpc_diagnosis_correct = mctpc[mctpc['diagnosis_correct'] == True]

mctpc_diagnosis_correct['time_NEW_to_FIN'] = mctpc_diagnosis_correct['FIN'] - mctpc_diagnosis_correct['NEW']

mctpc_diagnosis_incorrect = mctpc[mctpc['diagnosis_correct'] == False]

mctpc_diagnosis_incorrect['time_NEW_to_CHANGE_DIAGN'] = mctpc_diagnosis_correct['CHANGE DIAGN'] - mctpc_diagnosis_correct['NEW']
mctpc_diagnosis_incorrect['time_CHANGE_DIAGN_to_FIN'] = mctpc_diagnosis_correct['FIN'] - mctpc_diagnosis_correct['CHANGE DIAGN']

mctpc_document_not_ok = mctpc[mctpc['document_ok'] == False]

mctpc_document_not_ok['time_NEW_to_DELETE'] = mctpc_document_not_ok['DELETE'] - mctpc_document_not_ok['NEW']

mctpc_document_ok = mctpc[mctpc['document_ok'] == True]

mctpc_document_ok['time_FIN_to_RELEASE'] = mctpc_document_ok['RELEASE'] - mctpc_document_ok['FIN']
mctpc_document_ok['time_RELEASE_to_CODE_OK'] = mctpc_document_ok['CODE OK'] - mctpc_document_ok['RELEASE']
mctpc_document_ok['time_CODE_OK_to_BILLED'] = mctpc_document_ok['BILLED'] - mctpc_document_ok['CODE OK']