from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import pandas as pd
import numpy as np

# Hugging Face üzerindeki CSV'yi oku (gerekirse future kullanım için)
csv_url = "https://huggingface.co/datasets/eda12/relevant-detection-data/resolve/main/yeni_10.000.csv"
try:
    df = pd.read_csv(csv_url)
    print("✅ CSV başarıyla yüklendi.")
except Exception as e:
    print(f"❌ CSV yüklenemedi: {e}")
    df = None

# Embed modeli
bert_model = SentenceTransformer("paraphrase-MiniLM-L12-v2")

# Kategoriler
content_options = [
    "Biology", "Chemistry", "Computer", "Earth", "Medicine",
    "Nano", "Other", "Physics", "Space"
]

# Benzerliğe dayalı tahmin
def semantic_predict(context, category):
    context_embed = bert_model.encode([context])
    category_embed = bert_model.encode([category])
    similarity = cosine_similarity(context_embed, category_embed)[0][0]
    print(f"🔍 Benzerlik: {similarity:.3f}")
    return "relevant" if similarity >= 0.10 else "non-relevant"

# Gradio arayüzü
gr.Interface(
    fn=semantic_predict,
    inputs=[
        gr.Textbox(lines=6, label="Context (Text)"),
        gr.Dropdown(choices=content_options, label="Content (Category)")
    ],
    outputs=gr.Textbox(label="Relevance (relevant / non-relevant)"),
    title="Semantic Similarity Classifier",
    description="BERT ile anlam benzerliğine göre context ve content uyumu"
).launch(share=False, debug=True)


