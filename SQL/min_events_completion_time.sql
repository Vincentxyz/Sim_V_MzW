
CREATE VIEW min_event_completion_times AS
SELECT [_case_], [_event_], MIN([_completeTime_]) min_completion_time
  FROM [Hospital_Billing].[dbo].[Hospital_Billing_time_and_event_filtered]
  GROUP BY [_case_], [_event_]