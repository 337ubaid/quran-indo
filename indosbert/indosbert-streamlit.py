import streamlit as st
from sentence_transformers import SentenceTransformer, util
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import pickle
import torch
import pandas as pd

# LOAD ENV
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# INIT SESSION STATE
for key in ["username", "query", "query_is_submitted", "feedback_submitted", "results_model1", "results_model2"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["username", "query"] else False

# LOAD EMBEDDINGS & TEXTS
@st.cache_resource
def load_data():
    with open("indosbert/quran_id_full_embedding_indosbert.pkl", "rb") as f:
        emb1 = pickle.load(f, map_location=torch.device("cpu"))

    with open("indosbert/quran_id_full_embedding_indosbert.pkl", "rb") as f:
        emb2 = pickle.load(f, map_location=torch.device("cpu"))

    df = supabase.table("quran_id").select("id, indoText, suraId, verseID").execute()
    df = pd.DataFrame(df.data)
    return emb1, emb2, df

# GET SEARCH RESULTS
def search_results(query, model, embeddings, df, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0]

    results = []
    for i, hit in enumerate(hits):
        idx = hit["corpus_id"]
        row = df.iloc[idx]
        results.append({
            "id": int(row["id"]),
            "indoText": row["indoText"],
            "suraId": int(row["suraId"]),
            "verseID": int(row["verseID"]),
            "is_relevan": False,
        })
    return results

# DISPLAY SIDE-BY-SIDE
def show_comparison(results1, results2):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Model 1: `model_indosbert`")
        for i, res in enumerate(results1):
            st.write(f"**[{res['suraId']}:{res['verseID']}]** {res['indoText']}")
            results1[i]["is_relevan"] = st.checkbox("Relevan?", key=f"model1-{i}")
    with col2:
        st.markdown("### Model 2: `model_indosbert_finetune`")
        for i, res in enumerate(results2):
            st.write(f"**[{res['suraId']}:{res['verseID']}]** {res['indoText']}")
            results2[i]["is_relevan"] = st.checkbox("Relevan?", key=f"model2-{i}")

# SUBMIT TO SUPABASE
def add_feedback():
    supabase.table("feedback_ir").insert({
        "user": st.session_state.username,
        "query": st.session_state.query,
        "results_model1": st.session_state.results_model1,
        "results_model2": st.session_state.results_model2
    }).execute()
    st.session_state.feedback_submitted = True

# STEP 1: LOGIN
if not st.session_state.username:
    st.title("Pencarian Quran Bahasa Indonesia")
    username_input = st.text_input("Masukkan Username atau Inisial")
    username_input = username_input.strip()
    if st.button("Lanjut") and username_input != "":
        st.session_state.username = username_input
        st.rerun()

# STEP 2: QUERY
elif not st.session_state.query_is_submitted:
    st.title("Pencarian Quran Bahasa Indonesia")
    st.subheader(f"Hai, {st.session_state.username}!")
    query = st.text_input("Apa yang ingin kamu cari?")
    
    if st.button("Cari") and query != "":
        st.session_state.query = query
        st.session_state.query_is_submitted = True
        st.rerun()

# STEP 3: FEEDBACK COMPLETE
elif st.session_state.feedback_submitted:
    st.success("✅ Feedback berhasil dikirim!")
    st.write("🔄 Mau cari lagi atau keluar?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Cari lagi"):
            for key in ["query", "query_is_submitted", "feedback_submitted", "results_model1", "results_model2"]:
                st.session_state[key] = None if key == "query" else False
            st.rerun()
    with col2:
        if st.button("🚪 Keluar"):
            for key in st.session_state.keys():
                st.session_state[key] = None if key in ["username", "query"] else False
            st.rerun()

# STEP 4: SHOW RESULTS + FEEDBACK FORM
else:
    st.title("Hasil Pencarian")
    st.subheader(f"Oke, {st.session_state.username}!")
    st.markdown(f"### Hasil pencarian: `{st.session_state.query}`")

    # Load data & models
    emb1, emb2, df = load_data()
    model = SentenceTransformer("indobenchmark/indobert-base-p1")

    st.session_state.results_model1 = search_results(st.session_state.query, model, emb1, df)
    st.session_state.results_model2 = search_results(st.session_state.query, model, emb2, df)

    # Show results side-by-side
    show_comparison(st.session_state.results_model1, st.session_state.results_model2)

    if st.button("Submit Feedback"):
        add_feedback()
        st.rerun()
