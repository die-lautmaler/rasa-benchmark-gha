# rasa-benchmark-gha

Given a trained rasa model and a test dataset, this job completes if the accuracy is above a certain threshold, and fails otherwise.

### inputs
`model_path`: path that contains a trained rasa model .tar.gz file  
`duckling_url`: url to an active duckling server  
`test_data_path`: path to a test data set, csv or yaml  
`output_path`: file in which to store the results of the benchmark  
`gar_token`: your GAR token  
`gke_sa_key`: your GKE key  
`threshold`: 0.95  
