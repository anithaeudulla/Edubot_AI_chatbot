import os
import json
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import spacy
import numpy as np
import faiss
import google.generativeai as genai

# ------------------- 1️⃣ Folder Paths -------------------
PDF_FOLDER_PATH = r"D:\git hub\Edubot_AI_chatbot_for School student\jesc1dd"
TEXT_OUTPUT_FILE = r"D:\git hub\Edubot_AI_chatbot_for School student\cleaned_science_data.json"
PROCESSED_TEXT_FILE = r"D:\git hub\Edubot_AI_chatbot_for School student\processed_science_data.json"
EMBEDDINGS_FILE = r"D:\git hub\Edubot_AI_chatbot_for School student\spaCy_embeddings.json"
IMAGE_OUTPUT_FOLDER = r"D:\git hub\Edubot_AI_chatbot_for School student\extracted_images"
VECTOR_DB_PATH = r"D:\git hub\Edubot_AI_chatbot_for School student\vector_database.index"

# ------------------- 2️⃣ Extract Text from PDFs -------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)

data = {}
for file in os.listdir(PDF_FOLDER_PATH):
    if file.endswith(".pdf"):
        file_path = os.path.join(PDF_FOLDER_PATH, file)
        chapter_name = file.replace(".pdf", "")
        data[chapter_name] = extract_text_from_pdf(file_path)

with open(TEXT_OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("✅ Text extraction completed & saved as JSON!")

# ------------------- 3️⃣ Text Preprocessing -------------------
nltk.download("punkt")
nltk.download("stopwords")

def clean_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

with open(TEXT_OUTPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

for chapter in data:
    data[chapter] = clean_text(data[chapter])

with open(PROCESSED_TEXT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("✅ Text preprocessing completed & saved as JSON!")

# ------------------- 4️⃣ Convert to Embeddings -------------------
nlp = spacy.load("en_core_web_lg")

def get_embedding(text):
    return nlp(text).vector

with open(PROCESSED_TEXT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

embeddings = {chapter: get_embedding(data[chapter]).tolist() for chapter in data}

with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
    json.dump(embeddings, f, indent=4)

print("✅ Text converted to embeddings & saved as JSON using spaCy!")

# ------------------- 5️⃣ Extract Images -------------------
if not os.path.exists(IMAGE_OUTPUT_FOLDER):
    os.makedirs(IMAGE_OUTPUT_FOLDER)

image_count = 0
for filename in os.listdir(PDF_FOLDER_PATH):
    if filename.endswith(".pdf"):
        doc = fitz.open(os.path.join(PDF_FOLDER_PATH, filename))
        for page_num, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_filename = f"{filename}_page{page_num+1}_img{img_index+1}.png"
                with open(os.path.join(IMAGE_OUTPUT_FOLDER, img_filename), "wb") as img_file:
                    img_file.write(image_bytes)
                image_count += 1

print(f"✅ Extracted {image_count} images from all chapters.")

# ------------------- 6️⃣ Store in FAISS -------------------
with open(EMBEDDINGS_FILE, "r") as f:
    embeddings = json.load(f)

embedding_matrix = np.array(list(embeddings.values()), dtype='float32')
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)
faiss.write_index(index, VECTOR_DB_PATH)

print("✅ Embeddings stored in FAISS vector database!")

# ------------------- 7️⃣ Gemini-based QA System -------------------
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('models/gemini-1.5-flash')

with open(PROCESSED_TEXT_FILE, "r", encoding="utf-8") as f:
    processed_text = json.load(f)

chapter_names = list(embeddings.keys())
index = faiss.read_index(VECTOR_DB_PATH)

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

# ------------------- 8️⃣ Run the Chatbot -------------------
if __name__ == "__main__":
    question = input("❓ Ask your NCERT Science question: ")
    top_chaps = search_faiss(question)
    answer = query_gemini_with_context(question, top_chaps)
    print("\n💡 Answer:\n", answer)
