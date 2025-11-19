import streamlit as st
import pymongo
import google.generativeai as genai
import os

# =======================
# CONFIGURACIÓN
# =======================

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("❌ Faltan las variables de entorno GOOGLE_API_KEY o MONGODB_URI")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Conexión a MongoDB Atlas
client = pymongo.MongoClient(MONGODB_URI)
db = client["pdf_embeddings_db"]
collection = db["pdf_vectors"]

# =======================
# FUNCIONES
# =======================

def crear_embedding(texto):
    """Genera embedding de la pregunta"""
    model = "text-embedding-004"
    resp = genai.embed_content(model=model, content=texto)
    return resp["embedding"]

def buscar_similares(embedding, k=5):
    """
    Busca los documentos más similares en MongoDB Atlas.
    Requiere que el índice vectorial haya sido creado desde Atlas UI.
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    return list(collection.aggregate(pipeline))

def generar_respuesta(pregunta, contextos):
    """Usa Gemini para responder con contexto"""
    modelo = genai.GenerativeModel("gemini-flash-latest")
    contexto = "\n\n".join([c["texto"] for c in contextos])
    prompt = f"""
Eres un asistente experto. Usa el siguiente contexto para responder la pregunta del usuario.

Contexto:
{contexto}

Pregunta: {pregunta}

Responde de forma concisa y clara en español.
"""
    respuesta = modelo.generate_content(prompt)
    return respuesta.text

# =======================
# INTERFAZ STREAMLIT
# =======================

st.set_page_config(page_title="Chat PDF con MongoDB + Gemini", page_icon="💬")
st.title("💬 Chatbot de tu PDF (MongoDB + Gemini)")

if "historial" not in st.session_state:
    st.session_state.historial = []

pregunta = st.chat_input("Escribe tu pregunta sobre el PDF...")

if pregunta:
    with st.spinner("Buscando respuesta..."):
        emb = crear_embedding(pregunta)
        similares = buscar_similares(emb, k=5)

        if not similares:
            respuesta = "No encontré información relevante en el documento."
        else:
            respuesta = generar_respuesta(pregunta, similares)

        st.session_state.historial.append({"rol": "usuario", "texto": pregunta})
        st.session_state.historial.append({"rol": "bot", "texto": respuesta})

# Mostrar historial
for msg in st.session_state.historial:
    if msg["rol"] == "usuario":
        st.chat_message("user").write(msg["texto"])
    else:
        st.chat_message("assistant").write(msg["texto"])
