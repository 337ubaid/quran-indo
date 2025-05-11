test_id_retrieval = [1922, 3946, 6221,  169, 5948]

# LOAD LIBRARY
from sentence_transformers import SentenceTransformer
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import faiss

# LOAD CONNECTION TO SUPABASE
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# INISIALISASI SESSION STATE
for key in ["username", "query",  "model", "index", "query_is_submitted", "feedback_submitted", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "username" or key == "query" else False

# get ir
def get_results(query):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "search", "search-model")
    INDEX_PATH = os.path.join(BASE_DIR, "quran_embed_finetune.index")

    model = SentenceTransformer(MODEL_PATH)
    index= faiss.read_index(INDEX_PATH)
    st.session_state.model = model
    st.session_state.index = index

    query_vector = model.encode([query])
    top_5 = index.search(query_vector, 5)
    list_id = top_5[1].tolist()[0]

    results = supabase.table("quran_id").select("id, indoText, suraId, verseID").in_("id", list_id).execute()
    return results.data

def print_results(responses):
    # Header tabel
    col1, col2 , col3= st.columns([1, 5, 1])
    with col1:
        st.markdown("**Quran Surat**")  
    with col2:
        st.markdown("**Kalimat**")
    with col3:
        st.markdown("**Relevan?**")

    # Isi tabel
    for i, response in enumerate(responses):
        col1, col2, col3 = st.columns([1, 5, 1])
        with col1:
            st.write(f"{response['suraId']}, {response['verseID']}")
        with col2:
            st.write(response['indoText'])
        with col3:
            relevan = st.checkbox("", key=f"checkbox-{i}")
            response["is_relevan"] = relevan

# submit ir
def add_feedback():
    supabase.table('feedback_ir').insert({
        "user"  : st.session_state.username,
        "query" : st.session_state.query,
        "result": st.session_state.results
    }).execute()
    st.session_state.feedback_submitted = True
    
# isi username
if not st.session_state.username:
    st.title("Pencarian Quran Bahasa Indonesia")
    username_input = st.text_input("Masukkan Username atau Inisial")
    username_input = username_input.strip()
    if st.button("Lanjut") and username_input != "":
        st.session_state.username = username_input
        st.rerun()

# isi queryy
elif not st.session_state.query_is_submitted:
    st.title("Pencarian Quran Bahasa Indonesia")
    st.subheader(f"Hai, {st.session_state.username}!")
    query = st.text_input("Apa yang ingin kamu cari?")
    
    if st.button("Cari") and query != "":
        st.session_state.query = query
        st.session_state.query_is_submitted = True
        st.rerun()

# notif feedback
elif st.session_state.feedback_submitted:
    st.success("✅ Feedback berhasil dikirim!")
    st.write("🔄 Mau cari lagi atau keluar?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Cari lagi"):
            st.session_state.query_is_submitted = False
            st.session_state.feedback_submitted = False
            st.rerun()
    with col2:
        if st.button("🚪 Keluar"):
            for key in ["username", "query", "query_is_submitted", "feedback_submitted", "results"]:
                st.session_state[key] = None if key in ["username", "query"] else False
            st.rerun()

# feedbcak form
else :
    st.title("Hasil Pencarian")
    st.subheader(f"Oke, {st.session_state.username}!")
    st.markdown(f"### hasil pencarian: {st.session_state.query}")

    results = get_results(st.session_state.query)
    st.session_state.results = results
    print_results(results)
    if st.button("Submit Feedback"):
        add_feedback()
        st.rerun()