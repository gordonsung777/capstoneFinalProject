'''Build a dashboard (Streamlit) to display:
Real-time and forecasted KPIs'''
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate




print("\nLets set up the page first")
st.set_page_config(
    page_title="Digital Bank Key Performance Indicator Dashboard",
    layout="wide"
)

st.title("Digital Bank Observability Dashboard")

st.write(
    "==========Real-time Key Performance Indicator monitoring and short-term Key Performance Indicator forecasting (Task 4 a: Real-time and forecasted KPIs)==========="
)

@st.cache_data
def load_data():
    data = pd.read_csv("infrastructure_metrics.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp")
    return data

data = load_data()

#lets use the latest key performance indicator values
latest = data.iloc[-1]
previous = data.iloc[-2]

st.subheader("Current Key Performance Indicator Status")

col1, col2, col3 = st.columns(3)

col1.metric(
    "CPU Utilization",
    f"{latest['cpu_utilization']:.1f}%",
    f"{latest['cpu_utilization'] - previous['cpu_utilization']:.1f}%"
)

col2.metric(
    "Memory Usage",
    f"{latest['memory_usage']:.1f}%",
    f"{latest['memory_usage'] - previous['memory_usage']:.1f}%"
)

col3.metric(
    "DB Connection Pool",
    f"{latest['db_connection_pool_usage']:.1f}%",
    f"{latest['db_connection_pool_usage']-previous['db_connection_pool_usage']:.1f}%"
)

col4, col5, col6 = st.columns(3)

col4.metric(
    "Application Error Rate",
    f"{latest['application_error_rate']:.1f}"
)

col5.metric(
    "NEFT Success Rate",
    f"{latest['neft_success_rate']:.1f}%"
)

col6.metric(
    "Avg Processing Time",
    f"{latest['avg_transaction_processing_time_ms']:.0f} ms"
)

st.caption(
    f"Latest data timestamp: {latest['timestamp']}"
)

#Lets choose our Key performance indicator
st.subheader("Key Performance Indicator Trend and Forecast")

kpi_options = {
    "CPU Utilization": "cpu_utilization",
    "Memory Usage": "memory_usage",
    "DB Connection Pool": "db_connection_pool_usage",
    "Application Error Rate": "application_error_rate",
    "NEFT Success Rate": "neft_success_rate",
    "IMPS Success Rate": "imps_success_rate",
    "RTGS Success Rate": "rtgs_success_rate",
    "Transaction Processing Time": "avg_transaction_processing_time_ms",
    "Transaction Volume": "transaction_volume_per_hour"
}

selected_name = st.selectbox(
    "Select Key Performance Indicator",
    list(kpi_options.keys())
)

selected_kpi = kpi_options[
    selected_name
]

#Key performance indicator forecasting:

def forecast_kpi(data, column):
    #lets use the latest 60 observations
    recent = data.tail(60).copy()

    #we need to create time numbers such as 0, 1, 2, 3, ...
    X = np.arange(len(recent)).reshape(-1, 1)

    #the actual KPI values
    y = recent[column].values

    #create the model
    model = LinearRegression()

    #train the LinearRegression Model
    model.fit(
        X,
        y
    )

    # future 30 minutes

    future_X = np.arange(
        len(recent),
        len(recent) + 30
    ).reshape(-1, 1)

    #forecast predicted values
    predictions = model.predict(future_X)

    #lets create future timestamps
    future_times = pd.date_range(
        start=recent["timestamp"].iloc[-1] + pd.Timedelta(minutes=1),
        periods=30,
        freq="min"
    )

    forecast = pd.DataFrame({
        "timestamp": future_times,
        "Forecast": predictions
    })
    return recent, forecast

#lets execute and run the forecasting
recent, forecast = forecast_kpi(data, selected_kpi)

#the current and forecast predicted values

current_value = recent[selected_kpi].iloc[-1]
forecast_value = forecast["Forecast"].iloc[-1]
col1, col2 = st.columns(2)
col1.metric(f"Current{selected_name}", f"{current_value:.2f}")
col2.metric("30-Minute Forecast", f"{forecast_value:.2f}")

#the predicted forecast chart

actual_chart = recent[
    [
        "timestamp", selected_kpi
    ]
].copy()

actual_chart = actual_chart.rename(
    columns={
        selected_kpi: "Actual"
    }
)

actual_chart = actual_chart.set_index("timestamp")
forecast_chart = forecast.set_index("timestamp")

chart_data = pd.concat(
    [
        actual_chart,
        forecast_chart
    ],
    axis=1
)

st.line_chart(
    chart_data
)

#our recent KPI table

st.subheader("Recent KPI Records")

st.dataframe(
    data[
        [
            "timestamp",
            "cpu_utilization",
            "memory_usage",
            "db_connection_pool_usage",
            "application_error_rate",
            "neft_success_rate",
            "avg_transaction_processing_time_ms"
        ]
    ].tail(10),
    use_container_width=True
)


st.subheader("===============Log Classification Results (Task 4b)===================")
log_data = pd.read_csv("application_logs.csv")

#LEts keep only the ERROR and WARNING Logs
issue_logs = log_data[
    log_data["log_level"].isin(
        [
            "ERROR",
            "WARNING"
        ]
    )
].copy()

#Lets convert the Log Text into TG-IDF Values

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50
)

log_vectors = tfidf.fit_transform(
    issue_logs["log_message"]
)

#K-means log clustering technique

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

issue_logs["cluster"] = kmeans.fit_predict(
    log_vectors
)

#cluster 0 = Infrastructure Issues
#Cluster 1 = Business Key Performance Indicator issues

issue_logs["issue_type"] = issue_logs["cluster"].map({
    0: "Infrastructure Issues",
    1: "Business KPI Issues"
})

#Lets count the log types

log_counts = issue_logs["issue_type"].value_counts()

#Lets display the log counts

col1, col2 = st.columns(2)

col1.metric("Infrastructure Issues", log_counts.get("Infrastructure Issues", 0))
col2.metric("Business KPI Issues", log_counts.get("Business KPI Issues", 0))

#Log Classification Bar Chart
st.subheader("Log Issue Distribution")
st.bar_chart(log_counts)

#view classified logs
st.subheader("View Classified Logs")

selected_group = st.selectbox("Select Log Issue Group", ["All", "Infrastructure Issues", "Business KPI Issues"])

#Lets filter the logs based on selection
if selected_group == "All":
    filtered_logs = issue_logs
else:
    filtered_logs = issue_logs[
        issue_logs["issue_type"] == selected_group
    ]

#Display the logs
st.dataframe(
    filtered_logs[
        [
            "timestamp",
            "log_level",
            "log_message",
            "issue_type"
        ]
    ].head(50),
    use_container_width=True
)


#TASK 4C AI-generated incident analyses

st.subheader("===================AI-generated incident analyses (Task 4C)=====================")

st.write("Describe an incident below. The Streamlit Dashboard will retrieve the most releveant Root Cause Analysis document and use the current Key Performance Indicator"
         "values to generate an AI incident analysis.")

with open(
    "incident_knowledge_base.txt", "r", encoding="utf-8"
) as file:
    knowledge_text = file.read()

#Lets split the knowledge base into separate Root Cause Analysis Documents:
knowledge_documents =[]
documents = knowledge_text.split("================================================================================")
for document in documents:
    document = document.strip()

    if document:
        knowledge_documents.append(document)

#lets convert knowledge base into TF-IDF Values
knowledge_vectorizer = TfidfVectorizer(
    stop_words="english"
)

knowledge_vectors = knowledge_vectorizer.fit_transform(
    knowledge_documents
)

# have the user enter an incident issue

incident_query = st.text_area(
    "Please Describe your Incident:",

    placeholder=(
        "Example: NEFT transactions are failing with a gateway timeout"
    )
)

#Now we generate incident analysis report:

if st.button("Generate Incident Analysis"):
    if incident_query.strip() =="":
        st.warning("Please describe an incident first.")
    else:
        #Find the most relevant root cause analysis document
        
        query_vector = knowledge_vectorizer.transform([incident_query])

        similarity_scores = cosine_similarity(
            query_vector, 
            knowledge_vectors
        )[0]

        best_match = similarity_scores.argmax()

        retrieved_context = knowledge_documents[best_match]

        #LEts add the current Key Performance Indicator values:
        current_kpis = (
            f"CPU utilization: "
            f"{latest['cpu_utilization']:.1f}%\n"

            f"Memory Usage: "
            f"{latest['memory_usage']:.1f}%\n"

            f"Database Pool Usage: "
            f"{latest['db_connection_pool_usage']:.1f}%\n"

            f"Application Error Rate: "
            f"{latest['application_error_rate']:.1f}\n"

            f"NEFT Success Rate: "
            f"{latest['neft_success_rate']:.1f}%\n"

            f"IMPS Success Rate: "
            f"{latest['imps_success_rate']:.1f}%\n"

            f"RTGS Success Rate: "
            f"{latest['rtgs_success_rate']:.1f}%\n"
        )

        #Lets use OpenAI token on Vocareum
        api_key = "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"
        base_url = (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
        if api_key is None:
            st.error(
                "OPENAI_API_KEY was not found in Vocareum"
            )

        else:
            #If Vocareum provides OPENAI_MODEL, use it. Otherwise use this back up:
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            llm_settings = {
                "model": model_name,
                "api_key": api_key
            }

            #if Vocareum provides a custom OpenAI endpoint, LangChain will use it
            if base_url:
                llm_settings["base_url"] = base_url
            #Create the LangChain OpenAI model
            llm = ChatOpenAI(**llm_settings)

            #Lets Create the AI Prompt
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """
                    You are an AI incident-analysis assistant for a digital bank.

                    Yse the user's incident description, the current Key Performance Indicator values, and the retrieved Root Cause Analysis knowledge.

                    Please provide the following solutions/output/result:
                    
                    1) Likely Issue
                    2) Possible Root Cause
                    3) Recommended Action

                    Please keep the answer clear, practical and detailed.

                    If there is not enough information, say that more analysis and investigation needs to be done/required.

                    Current Key Performance Indicator values:
                    {current_kpis}

                    Retrieved Root Cause Analysis:
                    {context}
                    """
                ),
                (
                    "human",
                    "{question}"
                )
                
            ])

            #Lets connect the prompt to our OpenAI model
            chain = prompt | llm

            #Lets Ask the AI Model

            try:
                with st.spinner(
                    "Generating incident analysis..."
                ):

                    response = chain.invoke({
                        "current_kpis": current_kpis,
                        "context": retrieved_context,
                        "question": incident_query
                    })

                #Lets display AI Result

                st.subheader("AI Incident Analysis")

                st.write(response.content)

                #Lets Display the Retrieved Root Cause Analysis Document

                with st.expander("View Retrieved RCA Context"):
                    st.write(retrieved_context)

                    st.write(
                        "Similarity Score:",
                         round(
                             float(
                                 similarity_scores[best_match]
                             ),
                             3
                         )
                    )

            except Exception as error:
                st.error(
                    f"Unable to generate AI analysis: {error}"
                )


st.subheader("What our Streamlit Dashboard is demonstrating:\n")


st.write(
    """
    Our Streamlit dashboard was developed to provide a
    centralized view of current and forecasted infrastructure
    and business Key Performance Indicators.

    The latest timestamped observation is used to represent
    the current system state, while Linear Regression is
    applied to the most recent Key Performance Indicator
    observations to generate a short-term 30-minute forecast.

    You can interactively select metrics such as CPU
    Utilization, memory usage, database connection pool
    utilization, transaction success rates, processing time,
    and transaction volume to view both historical trends
    and predicted future behavior.

    Application ERROR and WARNING logs are converted into
    numerical features using TF-IDF and grouped using
    K-Means clustering. The logs are grouped into
    Infrastructure Issues and Business KPI Issues.

    The dashboard also includes an AI-generated incident
    analysis feature. A user can describe an incident, and 
    TF-IDF with cosine similarity retrieves the most relevant
    Root Cause Analysis document from the incident knowledge
    base. The retrieved information and current KPI values 
    are then passed to a LangChain OpenAI model to generate 
    the likely issue, possible root cause, and recommended 
    action.

    This supports proactive monitoring, incident investigation,
    and automated root cause analysis within the digital 
    banking system.
    """)
