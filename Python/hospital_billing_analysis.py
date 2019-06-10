
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load hospital_billing = hb
hb = pd.read_csv('C:/Users/vince_000/Documents/GitHub/Sim_V_MzW/Data/Hospital_Billing_time_and_event_filtered.csv')

# Show event log columns
hb.columns

# Adjust date format
hb['completeTime'] = hb['completeTime'].astype("datetime64") 

# We need to exclude the parient medical process, as we would have a backlog too high
hb = hb[hb.event.isin(['FIN', 'RELEASE', 'CODE OK', 'BILLED'])]

# Get Information about the dataset

## Timespan

hb.filter(['case']).groupby(hb['completeTime'].dt.year).count().plot(kind = "bar") 


## Filter on the most current cases year 2015 and 2016)

hb = hb[(hb['completeTime'].dt.year == 2015) |  (hb['completeTime'].dt.year == 2016)]

# Analysis: Count of events per user, also in comparison to Resource A and B
hb_release_code_ok_billed = hb[hb.event.isin(['RELEASE', 'CODE OK', 'BILLED'])]
event_user = hb_release_code_ok_billed.filter(['case','event','org:resource'])
event_user_pivot = event_user.pivot_table(values = 'case', index = 'org:resource', columns = 'event',  aggfunc='count', fill_value = 0)

ratio_release_res_a = event_user_pivot.loc['ResA'].RELEASE / event_user_pivot.RELEASE.sum()
ratio_code_ok_res_a = event_user_pivot.loc['ResA']['CODE OK'] / event_user_pivot['CODE OK'].sum()
ratio_billed_res_b = event_user_pivot.loc['ResB'].BILLED / event_user_pivot.BILLED.sum()

#events_per_user = event_user.pivot_table(index = 'org:resource', aggfunc = 'count', fill_value = 0)
#plt.hist(events_per_user['event'], bins = 100, density = True)


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,20):
    kmeans = KMeans(n_clusters = i,
                    init = 'k-means++',
                    max_iter = 300,
                    n_init = 10,
                    random_state = 0)
    kmeans.fit(event_user_pivot)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,20),wcss)

plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow method applied on user cluster groupings 2 to 19')
plt.show()

    
    
plt.plot(range(5,20),wcss[4:20])
plt.title('Elbow method applied on user cluster groupings 6 - 19')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means to the hospital event log
kmeans = KMeans(n_clusters = 6,
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


# creating a poisson distribution for the incoming events

hb_create_times = hb[hb['event'] == 'FIN'].filter(['completeTime']).sort_values(['completeTime'])
hb_timespan = hb_create_times.iloc[len(hb_create_times)-1,0] - hb_create_times.iloc[0,0]
mean_time_between_cases = hb_timespan / len(hb_create_times)
mean_hours_between_cases = mean_time_between_cases.total_seconds() / 60.0 / 60.0
print('Mean time between cases is: ' + str(round(mean_hours_between_cases,2)) + ' hours')

from scipy.stats import poisson
data_poisson = poisson.rvs(mu=mean_hours_between_cases, size=len(hb_create_times))
print(data_poisson)

import seaborn as sns
ax = sns.distplot(data_poisson,
                  bins=30,
                  kde=False,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Poisson Distribution', ylabel='Frequency')


# Calculating throughput times for the entering of distributions in Arena
min_completion_times = pd.read_csv('C:/Users/vince_000/Documents/GitHub/Sim_V_MzW/Data/Created/min_completion_times.txt')
min_completion_times['min_completion_time'] = min_completion_times['min_completion_time'].astype("datetime64") 

# Creating min_completion_times_per_case (mctpc)
mctpc = min_completion_times.pivot(values= 'min_completion_time', index = '_case_', columns = '_event_')

# Now comes the real calculation considering the process flow
mctpc['document_finalized'] = ~ pd.isnull(mctpc['FIN'])
mctpc['released'] = ~ pd.isnull(mctpc['RELEASE'])
mctpc['code_ok'] = ~ pd.isnull(mctpc['CODE OK'])
mctpc['billed'] = ~ pd.isnull(mctpc['BILLED'])

# Outliers with special process flos are not considered, so these are excluded
valid_entries =  (mctpc['document_finalized'] == True) & (mctpc['released'] == True) & \
                    (mctpc['code_ok'] == True) & (mctpc['billed'] == True) & (mctpc['RELEASE'] > mctpc['FIN']) & \
                    (mctpc['CODE OK'] > mctpc['RELEASE']) & (mctpc['BILLED'] > mctpc['CODE OK'])
mctpc = mctpc[valid_entries]

mctpc['time_FIN_to_RELEASE'] = mctpc['RELEASE'] - mctpc['FIN']
mctpc['time_RELEASE_to_CODE_OK'] = mctpc['CODE OK'] - mctpc['RELEASE']
mctpc['time_CODE_OK_to_BILLED'] = mctpc['BILLED'] - mctpc['CODE OK']
mctpc['time_FIN_to_BILLED'] = mctpc['BILLED'] - mctpc['FIN'] 

# get statistics from time values


def change_time_delta_to_days(df, event):
    df[event] = df[event].apply(func = lambda x : x.total_seconds() / 60.0 / 60.0 / 24.0 )

def change_time_delta_to_hours(df, event):
    df[event] = df[event].apply(func = lambda x : x.total_seconds() / 60.0 / 60.0 )


def calc_time_statistics(df, event):

    statistics_series = pd.Series({'event': event, 
                                   'min' : np.min(df[event]),         
                                   'mean' : np.mean(df[event]),
                                   'max': np.max(df[event])})
    return statistics_series



#plt.hist(list(mctpc_diagnosis_correct['time_NEW_to_FIN']),bins = 50, density = True)

change_time_delta_to_hours(mctpc,'time_FIN_to_RELEASE' )
change_time_delta_to_hours(mctpc,'time_RELEASE_to_CODE_OK' )
change_time_delta_to_hours(mctpc,'time_CODE_OK_to_BILLED' )
change_time_delta_to_days(mctpc, 'time_FIN_to_BILLED')

# Show histogram of entire billing process throughput time
plt.hist(mctpc['time_FIN_to_BILLED'],bins = 100, density = True)
plt.hist(mctpc['time_FIN_to_BILLED'],bins = 100, density = True, range =[0, 50])
plt.xlabel('Time of patient finalization to billed status')
plt.ylabel('Frequency density')
plt.title('Throughput time of medical billing proces')


df_events = [[mctpc,'time_FIN_to_RELEASE' ],
             [mctpc,'time_RELEASE_to_CODE_OK'],
             [mctpc, 'time_CODE_OK_to_BILLED'],
             [mctpc, 'time_FIN_to_BILLED']]    


# Get time statistics per event

df_time_statistics = pd.DataFrame(columns = ['event', 'min', 'mean', 'max'])
for i in range(0, len(df_events)):
#    change_time_delta_to_days(df_events[i][0], df_events[i][1])
    df_time_statistics = df_time_statistics.append(calc_time_statistics(df_events[i][0], df_events[i][1]), ignore_index = True)
    
# merge tables
mctpc = mctpc.filter(['time_FIN_to_RELEASE', 'time_RELEASE_to_CODE_OK', 'time_CODE_OK_to_BILLED'])

plt.hist(list(mctpc['time_FIN_to_RELEASE']), bins = 100, density = True)


hb.set_index(keys = 'case', drop = True, inplace = True)

#Create hbcue = hb_cluster_user_events
hbcue = hb.filter(['event', 'org:resource', 'cluster'])
hbcue_release = hbcue[hbcue['event'] == 'RELEASE']
hbcue_code_ok = hbcue[hbcue['event'] == 'CODE OK']
hbcue_billed = hbcue[hbcue['event'] == 'BILLED']


# Calculate time statistics per event and show the distributions in histograms

## FIN to RELEASE 
hbcue_release = hbcue_release.join(mctpc.filter(['time_FIN_to_RELEASE']),how = 'inner')
plt.hist(list(hbcue_release['time_FIN_to_RELEASE']),bins = 50, density = True)
plt.hist(list(hbcue_release['time_FIN_to_RELEASE']),bins = 50, density = True, range = [0,50]) 

plt.hist(list(hbcue_release[hbcue_release['org:resource'] != 'ResA']['time_FIN_to_RELEASE']),bins = 50, density = True)
plt.hist(list(hbcue_release[hbcue_release['org:resource'] != 'ResA']['time_FIN_to_RELEASE']),bins = 50, density = True, range = [0,50])

min_FIN_to_RELEASE = hbcue_release.filter(['time_FIN_to_RELEASE']).min()
min_FIN_to_RELEASE.columns = ['min_FIN_to_RELEASE']
median_FIN_to_RELEASE = hbcue_release.filter(['time_FIN_to_RELEASE']).median()
median_FIN_to_RELEASE.columns = ['median_FIN_to_RELEASE']
max_FIN_to_RELEASE = hbcue_release.filter(['time_FIN_to_RELEASE']).max()
max_FIN_to_RELEASE.columns = ['max_FIN_to_RELEASE']

## RELEASE to CODE OK 
hbcue_code_ok = hbcue_code_ok.join(mctpc.filter(['time_RELEASE_to_CODE_OK']),how = 'inner')
plt.hist(list(hbcue_code_ok['time_RELEASE_to_CODE_OK']),bins = 50, density = True)
plt.hist(list(hbcue_code_ok['time_RELEASE_to_CODE_OK']),bins = 50, density = True, range = [0,300])

min_RELEASE_to_CODE_OK = hbcue_code_ok.filter(['time_RELEASE_to_CODE_OK']).min()
min_RELEASE_to_CODE_OK.columns = ['min_RELEASE_to_CODE_OK']
median_RELEASE_to_CODE_OK = hbcue_code_ok.filter(['time_RELEASE_to_CODE_OK']).median()
median_RELEASE_to_CODE_OK.columns = ['median_RELEASE_to_CODE_OK']
max_RELEASE_to_CODE_OK = hbcue_code_ok.filter(['time_RELEASE_to_CODE_OK']).max()
max_RELEASE_to_CODE_OK.columns = ['max_RELEASE_to_CODE_OK']

## CODE OK to BILLED 
hbcue_billed = hbcue_billed.join(mctpc.filter(['time_CODE_OK_to_BILLED']),how = 'inner')
plt.hist(list(hbcue_billed['time_CODE_OK_to_BILLED']),bins = 50, density = True)
plt.hist(list(hbcue_billed['time_CODE_OK_to_BILLED']),bins = 50, density = True, range = [0,50])

min_CODE_OK_to_BILLED = hbcue_billed.filter(['time_CODE_OK_to_BILLED']).min()
min_CODE_OK_to_BILLED.columns = ['min_FIN_to_RELEASE']
median_CODE_OK_to_BILLED = hbcue_billed.filter(['time_CODE_OK_to_BILLED']).median()
median_CODE_OK_to_BILLED.columns = ['median_FIN_to_RELEASE']
max_CODE_OK_to_BILLED = hbcue_billed.filter(['time_CODE_OK_to_BILLED']).max()
max_CODE_OK_to_BILLED.columns = ['max_CODE_OK_to_BILLED']

# Get availability per user
total_times = hbcue_release.filter(['org:resource', 'time_FIN_to_RELEASE']).groupby(['org:resource']).sum()
total_times['time_RELEASE_to_CODE_OK'] = hbcue_code_ok.filter(['org:resource', 'time_RELEASE_to_CODE_OK']).groupby(['org:resource']).sum()
total_times['time_CODE_OK_to_BILLED'] = hbcue_billed.filter(['org:resource', 'time_CODE_OK_to_BILLED']).groupby(['org:resource']).sum()
total_times.fillna(0, inplace = True)
time_span_in_h = hb_timespan.total_seconds() / 60.0 / 60.0 
total_times['availability'] = (total_times['time_FIN_to_RELEASE'] + total_times['time_RELEASE_to_CODE_OK'] + total_times['time_CODE_OK_to_BILLED']) / time_span_in_h

time_stats = pd.DataFrame({'min_FIN_to_RELEASE' : min_FIN_to_RELEASE,
                           'median_FIN_to_RELEASE' : median_FIN_to_RELEASE,
                           'max_FIN_to_RELEASE' : max_FIN_to_RELEASE,
                           'min_RELEASE_to_CODE_OK' : min_RELEASE_to_CODE_OK,
                           'median_RELEASE_to_CODE_OK' : median_RELEASE_to_CODE_OK,
                           'max_RELEASE_to_CODE_OK' : max_RELEASE_to_CODE_OK,
                           'min_CODE_OK_to_BILLED' : min_CODE_OK_to_BILLED,
                           'median_CODE_OK_to_BILLED' : median_CODE_OK_to_BILLED,
                           'max_CODE_OK_to_BILLED' : max_CODE_OK_to_BILLED})
