import streamlit as st
from sentence_transformers import SentenceTransformer, util
from supabase import create_client, Client
from dotenv import load_dotenv
import os
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

st.set_page_config(layout="wide")

# LOAD EMBEDDINGS & TEXTS
@st.cache_resource
def load_data():
    model1 = SentenceTransformer("denaya/indoSBERT-large")
    model2 = SentenceTransformer("devvevan/indo-sbert-finetuned-qa")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(BASE_DIR, "quran_id_full_embedding_indosbert.pt"), "rb") as f:
        emb1 = torch.load(f, map_location=torch.device('cpu'))

    with open(os.path.join(BASE_DIR, "quran_context_embedding_indosbert_finetune.pt"), "rb") as f:
        emb2 = torch.load(f, map_location=torch.device('cpu'))
    
    # Loop untuk ambil data bertahap
    # Parameter
    limit = 1000
    offset = 0
    all_data = []

    while True:
        response = supabase.table("quran_id") \
            .select("id, indoText, suraId, verseID") \
            .range(offset, offset + limit - 1) \
            .execute()
        
        batch = response.data
        if not batch:
            break
        
        all_data.extend(batch)
        offset += limit

        if offset >= 6500:
            break

    # Konversi ke DataFrame
    df = pd.DataFrame(all_data)
    return emb1, emb2, df, model1, model2

# GET SEARCH RESULTS
def search_results(query, model, embeddings, df, top_k=15):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0]

    results = []
    for hit in hits:
        idx = hit['corpus_id']
        score = hit['score']
        results.append({
            "suraId": int(df.iloc[idx]['suraId']),
            "verseID": int(df.iloc[idx]['verseID']),
            "indoText": df.iloc[idx]['indoText'],
            "score": float(score)
            })
    return results

# DISPLAY SIDE-BY-SIDE
def show_comparison(results1, results2):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='text-align: center;'>Model 1</h3>", unsafe_allow_html=True)
        for i, res in enumerate(results1):
            sub_col11, sub_col12, sub_col13 = st.columns([1, 5, 1])
            with sub_col11:
                st.write(f"{res['suraId']}, {res['verseID']}")
            with sub_col12:
                st.write(res['indoText'])
            with sub_col13:
                res['is_relevan'] = st.checkbox("", key=f"model1-{i}")  # hanya simpan di session_state

    with col2:
        st.markdown("<h3 style='text-align: center;'>Model 2</h3>", unsafe_allow_html=True)
        for i, res in enumerate(results2):
            sub_col21, sub_col22, sub_col23 = st.columns([1, 5, 1])
            with sub_col21:
                st.write(f"{res['suraId']}, {res['verseID']}")
            with sub_col22:
                st.write(res['indoText'])
            with sub_col23:
                res['is_relevan'] = st.checkbox("", key=f"model2-{i}")

# SUBMIT TO SUPABASE
def add_feedback():
    supabase.table("feedback_ir").insert({
        "user": st.session_state.username,
        "query": st.session_state.query,
        "result1": st.session_state.results_model1,
        "result2": st.session_state.results_model2
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
    emb1, emb2, df, model1, model2 = load_data()
    

    st.session_state.results_model1 = search_results(st.session_state.query, model1, emb1, df)
    st.session_state.results_model2 = search_results(st.session_state.query, model2, emb2, df)

    # Show results side-by-side
    with st.form("feedback_form"):
        show_comparison(st.session_state.results_model1, st.session_state.results_model2)
        st.markdown("---")
        st.markdown("---")

        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            add_feedback()
            st.rerun()


