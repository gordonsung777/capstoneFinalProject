'''Task 4b) Create backend endpoints using FastAPI for model inference and log insights'''

from pathlib import Path

import pandas as pd
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

#LEts Create FastAPI App

app = FastAPI(
    title="Digital Bank Observability API",
    description="This is our Backend Endpoints for incident prediction and log insights analysis"
)

#Lets train incident Criticality Model

incident_data = pd.read_csv(
    "incident_reports.csv"
)

incident_features = [
    "cpu_utilization",
    "memory_usage",
    "db_connection_pool_usage",
    "network_throughput_mbps",
    "application_error_rate",
    "neft_success_rate",
    "imps_success_rate",
    "rtgs_success_rate",
    "avg_transaction_processing_time_ms"
]

X_incident = incident_data[incident_features]
y_incident = incident_data["is_critical"]

incident_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

incident_model.fit(
    X_incident,
    y_incident
)

#3) Train log clustering model

log_data = pd.read_csv(
    "application_logs.csv"
)

issue_logs = log_data[
    log_data["log_level"].isin(
        ["ERROR", "WARNING"]
    )
].copy()

tfidf = TfidfVectorizer(
    stop_words = "english",
    max_features =50
)

log_vectors = tfidf.fit_transform(
    issue_logs["log_message"]
)

log_model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

issue_logs["cluster"] = log_model.fit_predict(log_vectors)


#automatically name the 2 log clusters we created

terms = np.array(tfidf.get_feature_names_out())

infrastructure_words = {
    "cpu",
    "memory",
    "database",
    "disk",
    "network",
    "server",
    "srv",
    "latency",
    "bottleneck",
    "oom",
    "connection",
    "pool",
    "container",
    "restarting"
}

business_words = {
    "transaction",
    "neft",
    "imps",
    "rtgs",
    "payment",
    "gateway",
    "failed",
    "duplicate",
    "beneficiary"
}

cluster_names = {}

for cluster_number in range(2):
  top_indexes = (log_model.cluster_centers_[cluster_number].argsort()[::-1][:15])
  top_words = set(terms[top_indexes])
  infrastructure_score = len(top_words.intersection(infrastructure_words))
  business_score = len(top_words.intersection(business_words))
  if infrastructure_score > business_score:
    cluster_names[cluster_number] = "Infrastructure Issue"
  else:
    cluster_names[cluster_number] = "Business KPI Issue"

issue_logs["issue_type"] = issue_logs["cluster"].map(cluster_names)


#lets define json input formats 

class IncidentInput(BaseModel):
  cpu_utilization: float
  memory_usage: float
  db_connection_pool_usage: float
  network_throughput_mbps: float
  application_error_rate: float
  neft_success_rate: float
  imps_success_rate: float
  rtgs_success_rate: float
  avg_transaction_processing_time_ms: float

class LogInput(BaseModel): 
  log_message: str

#our home endpoint

@app.get("/")
def home():
  return {
      "message": "Digital Bank Observability API is running"
  }

#Incident Criticality Endpoint
@app.post("/predict-criticality")
def predict_criticality(incident: IncidentInput):

  values = [[
      incident.cpu_utilization,
      incident.memory_usage,
      incident.db_connection_pool_usage,
      incident.network_throughput_mbps,
      incident.application_error_rate,
      incident.neft_success_rate,
      incident.imps_success_rate,
      incident.rtgs_success_rate,
      incident.avg_transaction_processing_time_ms
  ]]

  prediction = incident_model.predict(values)[0]
  probability = incident_model.predict_proba(values)[0][1]
  
  if prediction == 1:
    result = "Critical"
  else: 
    result = "Non-Critical"
  
  return {
      "prediction":result,
      "critical_probability": round(float(probability), 3)
  }

#Lets Classify a New Log message

@app.post("/log-insight")
def log_insight(log: LogInput):
  vector = tfidf.transform([log.log_message])
  cluster = log_model.predict(vector)[0]
  issue_type = cluster_names[cluster]
  return{
      "log_message": log.log_message,
      "issue_type": issue_type,
      "cluster": int(cluster)
  }

#the log summary endpoint
@app.get("/log-insights/summary")
def log_summary():
  counts = issue_logs["issue_type"].value_counts()
  return{
      "total_issue_logs": 
        int(len(issue_logs)), 
      "infrastructure_issues": 
        int(counts.get(
            "Infrastructure Issue",
            0
        )
      ),
      "business_kpi_issues":
        int(
            counts.get(
                "Business KPI Issue", 0
            )
        )
      }

