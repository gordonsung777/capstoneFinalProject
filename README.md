Business Introduction
In today’s fast-paced digital banking environment, financial institutions process millions of transactions daily across channels such as NEFT, IMPS, and RTGS. As systems grow in complexity, banks face challenges in maintaining seamless operations and proactively managing incidents.

Traditional monitoring tools often work in silos—separating business metrics, application logs, and infrastructure data. This fragmentation delays root cause identification and increases downtime during service disruptions.

An AI-driven observability platform can unify monitoring, prediction, and automation. By integrating machine learning, natural language processing, and agentic AI, financial institutions can predict failures, detect anomalies, and automatically analyze incidents before they affect customers.

Problem statement

Design and implement an end-to-end intelligent observability and predictive analytics system for a digital bank.

The system should:

Monitor business KPIs
Track infrastructure KPIs
Predict performance degradation and forecast key metrics using machine learning
Classify and cluster logs to identify issue types
Provide AI-generated root cause analysis through an integrated agentic AI interface
Deliver all insights through an interactive dashboard
The goal is to build a solution that shifts banking operations from reactive monitoring to predictive intelligence.

Input dataset: DatasetLinks to an external site.

Tasks

Task 1: Data preparation and exploration

Load datasets, including transactions_fraud_train.csv, infrastructure_metrics.csv, and application_logs.csv
Clean the data by handling missing values and standardizing data types
Explore distributions of key business KPIs, such as success rates and processing times
Analyze infrastructure metrics, including CPU utilization, memory usage, and error rates
Correlate transaction performance with infrastructure load
Visualize patterns to identify system bottlenecks and anomalies
 Task 2: Machine learning and predictive modeling

Implement time-series forecasting models to predict transaction success rates, API latency, and resource utilization
Build a classification model to predict incident criticality based on KPIs
Apply anomaly detection to identify unusual metric behavior
Evaluate model performance using metrics such as RMSE, MAE, Accuracy, and F1-score
Save trained models for deployment
 Task 3: AI-Driven incident analysis and root cause automation

Preprocess log data using NLP techniques such as TF-IDF or word embeddings
Apply unsupervised clustering to group logs into infrastructure issues and business KPI issues
Build a knowledge base from historical incidents and documentation
Integrate a retrieval-augmented generation (RAG) pipeline to fetch relevant incidents and documents
Implement an AI assistant using LangGraph or LangChain to answer queries
Generate AI-based root cause summaries and recommendations
Task 4: Visualization and deployment

Build a dashboard (Streamlit) to display:
Real-time and forecasted KPIs
Log classification results
AI-generated incident analyses
Create backend endpoints using FastAPI for model inference and log insights
Containerize components with Docker and orchestrate using Docker Compose
Deploy and test the system locally or on the cloud
