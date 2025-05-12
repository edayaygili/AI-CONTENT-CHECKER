import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Modeli yükle
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L12-v2")

model = load_model()

# Arayüz başlığı
st.title("🔍 AI Content Relevance Checker")
st.markdown("BERT modeliyle context ve kategori uyumunu değerlendirir.")

# Kullanıcıdan giriş al
context = st.text_area("Context (Text)", height=200)
category = st.selectbox(
    "Content Category",
    ["Biology", "Chemistry", "Computer", "Earth", "Medicine", "Nano", "Other", "Physics", "Space"]
)

if st.button("Check Relevance"):
    context_embed = model.encode([context])
    category_embed = model.encode([category])
    similarity = cosine_similarity(context_embed, category_embed)[0][0]
    
    st.markdown(f"**Similarity Score:** {similarity:.3f}")
    
    if similarity >= 0.10:
        st.success("✅ Relevant")
    else:
        st.error("❌ Non-Relevant")



