import streamlit as st
import json
import numpy as np
import faiss
import spacy
import google.generativeai as genai

# ------------------- Configuration -------------------
PDF_PROCESSED_TEXT_FILE = r"D:\\git hub\\Edubot_AI_chatbot_for School student\\processed_science_data.json"
EMBEDDINGS_FILE = r"D:\\git hub\\Edubot_AI_chatbot_for School student\\spaCy_embeddings.json"
VECTOR_DB_PATH = r"D:\\git hub\\Edubot_AI_chatbot_for School student\\vector_database.index"

# ------------------- UI: Logo and Title -------------------
st.title("📘 Class 10 Science Chatbot (NCERT-based)")

# ------------------- Load Data -------------------
@st.cache_resource
def load_resources():
    nlp_model = spacy.load("en_core_web_lg")
    with open(PDF_PROCESSED_TEXT_FILE, "r", encoding="utf-8") as f:
        text_data = json.load(f)
    with open(EMBEDDINGS_FILE, "r") as f:
        embeddings = json.load(f)
    faiss_index = faiss.read_index(VECTOR_DB_PATH)
    chapter_names = list(embeddings.keys())
    return nlp_model, text_data, embeddings, faiss_index, chapter_names

nlp, processed_text, embeddings, index, chapter_names = load_resources()

# ------------------- Gemini Setup -------------------
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('models/gemini-1.5-flash')

# ------------------- Embedding & Search -------------------
def get_embedding(text):
    return nlp(text).vector

def search_faiss(query, top_k=3):
    query_vector = np.array([get_embedding(query)], dtype='float32')
    distances, indices = index.search(query_vector, top_k)
    return [chapter_names[i] for i in indices[0]]

def query_gemini_with_context(query, top_chapters):
    context = "\n\n".join([f"{ch}:\n{processed_text[ch]}" for ch in top_chapters])
    prompt = f"""You are a helpful AI tutor for Class 10 Science students.
Use the following NCERT content to answer the question accurately and simply.

Context:
{context}

Question: {query}

Answer in simple terms as per NCERT syllabus:"""
    response = model.generate_content(prompt)
    return response.text

# ------------------- Streamlit UI -------------------
user_query = st.text_input("❓ Ask your NCERT Science question:", placeholder="e.g., What is photosynthesis?")

if user_query:
    with st.spinner("Searching and generating answer..."):
        top_chaps = search_faiss(user_query)
        answer = query_gemini_with_context(user_query, top_chaps)
    st.markdown("---")
    st.subheader("💡 Answer")
    st.write(answer)


#mysql connection
from sqlalchemy import create_engine
import pandas as pd
from urllib.parse import quote  # ✅ Needed to encode special characters
from datetime import datetime
import streamlit as st

# Credentials and safe password handling
user = 'root'  # Database username
pw = quote('Anitha@1')  # Encode password
db = 'edubot'  # Database name
# Create MySQL connection engine
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Load Queries from DB
@st.cache_data
def load_logs():
    return pd.read_sql("SELECT * FROM student_queries ORDER BY timestamp DESC", engine)

# Streamlit UI
st.title("📚 Edubot Query Logs")

logs_df = load_logs()
st.dataframe(logs_df, use_container_width=True)

# Optional Analytics
st.subheader("📈 Most Asked Chapters")
chapter_freq = logs_df["matched_chapters"].str.split(", ").explode().value_counts()
st.bar_chart(chapter_freq)

#Logging Function:
def log_query_to_db(query, top_chapters):
    log_data = pd.DataFrame([{
        "query": query,
        "matched_chapters": ", ".join(top_chapters),
        "timestamp": datetime.now()
    }])
    log_data.to_sql("student_queries", con=engine, if_exists='append', index=False)


 # Call the Function After Showing the Answer:
if user_query:
    with st.spinner("Searching and generating answer..."):
        top_chaps = search_faiss(user_query)
        answer = query_gemini_with_context(user_query, top_chaps)

    st.markdown("---")
    st.subheader("💡 Answer")
    st.write(answer)

    # Log query to MySQL
    log_query_to_db(user_query, top_chaps)

