

import pandas as pd
import numpy as np

hospital_billing = pd.read_csv('C:/Users/vince_000/Documents/GitHub/Sim_V_MzW/Data/Hospital_Billing.csv')

hospital_billing.columns

# Analysis: Count of events per user

event_user = hospital_billing.filter(['case','event','org:resource'])
event_user_pivot = event_user.pivot_table(values = 'case', index = 'org:resource', columns = 'event',  aggfunc='count', fill_value = 0)

# Active hours per user

