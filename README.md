# rasa-benchmark-gha

Given a trained rasa model and a test dataset, this job completes if the accuracy is above a certain threshold, and fails otherwise.

### inputs
`model_path`: path that contains a trained rasa model .tar.gz file  
`duckling_url`: url to an active duckling server  
`test_data_path`: path to a test data file or folder. If a folder is given, the path needs to end in csv or yaml folder (quirk of benchmarking tool needs to be supported)  
`output_path`: file in which to store the results of the benchmark  
`gar_token`: your GAR token  
`gke_sa_key`: your GKE key  
`threshold`: 0.95  

### usage
```
- name: benchmark the rasa model
        uses: die-lautmaler/rasa-benchmark-gha@v1
        with:
          model_path: './bots/logo-demo-nlu/models'  # Make sure this points to your model directory
          duckling_url: ${{ secrets.DUCKLING_URL }}
          test_data_path: 'bots/logo-demo-nlu/data/test.yml'  # Adjust this path as needed
          output_path: './benchmark_results.json'
          gar_token: ${{ env.GAR_TOKEN }}
          gke_sa_key: '${{ secrets.GKE_CCA_SA_KEY }}'
          threshold: 0.95
```

### todo
- currently the output_path is not used and no report is exported. working on generating the report.
