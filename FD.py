import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import httpcore

# Initialize Ollama LLM model
try:
    model = OllamaLLM(model="llava")
except Exception as e:
    st.error("Failed to initialize OllamaLLM. Ensure the server is running. Error: " + str(e))
    model = None

# Template with conversation history
template = """
Answer the question below.

Here is the conversation history:
{context}

Question: {question}

Answer:
"""

if model:
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

# Title and Description
st.title("Fixed Deposit Insights & Analysis")
st.write("Analyze fixed deposit data with insights, visualizations, and sentiment analysis tailored to General and Senior Citizens.")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file with fixed deposit data", type="csv")


# LLM Chat Section
st.sidebar.header("LLM Chat")
if model:
    user_query = st.text_input("Ask a question about fixed deposit data:")
    if user_query:
        try:
            result = chain.invoke({"context": "", "question": user_query}).strip()
            st.write("### Response:")
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.sidebar.warning("LLM is not available. Please check your server setup.")


# Placeholder function for sentiment analysis
def perform_sentiment_analysis(df):
    positive_count = (df['Maturity Amount'] > df['Principal Amount']).sum()
    negative_count = (df['Maturity Amount'] <= df['Principal Amount']).sum()
    insights = {
        "Positive Impact": positive_count,
        "Negative Impact": negative_count,
        "Overall Sentiment": "Positive" if positive_count > negative_count else "Negative",
    }
    return insights

# Function to filter data by citizen type
def filter_by_citizen_type(df, citizen_type):
    if citizen_type == "General Citizen":
        return df[(df['Age'] >= 18) & (df['Age'] <= 54)]
    elif citizen_type == "Senior Citizen":
        return df[df['Age'] >= 55]

# Function to calculate insights and map interest rates based on tenure
def calculate_insights(df, citizen_type):
    rates = {
        "General Citizen": [(1, 1.25, 6.70), (1.25, 1.5, 7.25), (1.5, 2, 7.25)],
        "Senior Citizen": [(1, 1.25, 7.20), (1.25, 1.5, 7.80), (1.5, 2, 7.75)]
    }
    def map_interest_rate(tenure):
        for lower, upper, rate in rates[citizen_type]:
            if lower <= tenure <= upper:
                return rate
        return None
    df['Mapped Interest Rate (%)'] = df['Tenure (Years)'].apply(map_interest_rate)
    summary = {
        "Total Customers": len(df),
        "Total Principal Amount": f"${df['Principal Amount'].sum():,.2f}",
        "Total Maturity Amount": f"${df['Maturity Amount'].sum():,.2f}",
        "Average Interest Rate (%)": f"{df['Mapped Interest Rate (%)'].mean():,.2f}%",
    }
    return df, summary

# Function to generate visualizations
def generate_visualizations(df, chart_type):
    if chart_type == "Principal vs Maturity Amount Bar Plot":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='FD ID', y='Principal Amount', data=df, color='blue', label='Principal')
        sns.barplot(x='FD ID', y='Maturity Amount', data=df, color='green', label='Maturity')
        ax.set_title('Principal vs Maturity Amount')
        ax.legend()
        return fig
    elif chart_type == "Mapped Interest Rate Distribution":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Mapped Interest Rate (%)'], kde=True, color='orange', bins=10)
        ax.set_title('Mapped Interest Rate Distribution')
        return fig
    elif chart_type == "Tenure vs Maturity Amount Scatter Plot":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Tenure (Years)', y='Maturity Amount', data=df, hue='Mapped Interest Rate (%)', palette='cool')
        ax.set_title('Tenure vs Maturity Amount')
        return fig

# File processing and data display
if uploaded_file:
    with st.spinner("Processing data..."):
        time.sleep(2)
    try:
        data = pd.read_csv(uploaded_file)
        required_columns = {'Customer ID', 'Age', 'Principal Amount', 'Maturity Amount', 'Tenure (Years)', 'FD ID'}
        if not required_columns.issubset(data.columns):
            st.error(f"The uploaded file must contain these columns: {', '.join(required_columns)}")
        else:
            citizen_type = st.sidebar.radio("Choose a citizen type:", ["General Citizen", "Senior Citizen"])
            filtered_data = filter_by_citizen_type(data, citizen_type)
            if filtered_data.empty:
                st.error(f"No data available for {citizen_type}.")
            else:
                processed_data, summary = calculate_insights(filtered_data, citizen_type)
                st.header(f"{citizen_type} Insights")
                for key, value in summary.items():
                    st.metric(label=key, value=value)
                sentiment_insights = perform_sentiment_analysis(filtered_data)
                st.subheader("Sentiment Analysis")
                for key, value in sentiment_insights.items():
                    st.write(f"{key}: {value}")
                st.header("Processed Data")
                st.dataframe(processed_data)
                chart_type = st.sidebar.radio("Choose a chart type:", [
                    "Principal vs Maturity Amount Bar Plot",
                    "Mapped Interest Rate Distribution",
                    "Tenure vs Maturity Amount Scatter Plot"
                ])
                st.header("Visualizations")
                fig = generate_visualizations(processed_data, chart_type)
                st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a CSV file to get started.")

