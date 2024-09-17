init:
	pip3 install -r requirements.txt
	mkdir -p data/csv/
	mkdir -p data/results/json/
	mkdir -p data/results/csv/
	mkdir -p data/reports/

help:
	echo "-"

fetch:
	export GOOGLE_APPLICATION_CREDENTIALS="./keys/botify-basic-dev.json"
	curl -X GET -H "Authorization: Bearer  ya29.c.Kp0B8QcwU80a1sFTldg_wiqA_dQaGC35vBKsueyxxtUtanHZ-Yh-jDazqq_WoYWQk3X7IOr01HkrtGEoQ-x2knj2Eb1ApSH4CyBOg57mTFVS4Igqij2If5C9tdqGxfG-_NwxOQmKkLgBmG9R1QV8Gxd5RxipdeLJl15HuO65R1cOCw0oHzh3QJFYlnPkYh9VA9A9i3CGbJ_wgAz7TofELw" https://dialogflow.googleapis.com/v2/projects/botify-basic-dev/agent/validationResult?languageCode=de

.PHONY: init tests fetch
#$(gcloud auth application-default print-access-token)