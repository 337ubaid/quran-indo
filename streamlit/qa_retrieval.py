test_id_retrieval = [1922, 3946, 6221,  169, 5948]

# LOAD LIBRARY
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv
import os
from datetime import datetime
import faiss

# LOAD CONNECTION TO SUPABASE
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# CEK KONDISI SESSION STATE
if "username" not in st.session_state:
    st.session_state.username = None

if "query_is_submitted" not in st.session_state:
    st.session_state.query_is_submitted = False


# get ir
def get_results(query):
    model_finetune = model = SentenceTransformer('../sbert/search/search-model')
    index_finetune = faiss.read_index("quran_embed_finetune.index")
    results=search(query, top_k=5, index=index_finetune, model=model_finetune)
    results = supabase.table("quran_id").select("id, indoText, suraId, verseID").in_("id", test_id_retrieval).execute()
    return results.data

def print_results(response):
    # Header tabel
    col1, col2 , col3= st.columns([1, 5, 1])
    with col1:
        st.markdown("**Quran Surat**")  
    with col2:
        st.markdown("**Kalimat**")
    with col3:
        st.markdown("**Relevan?**")

    # Isi tabel
    for i, response in enumerate(response):
        col1, col2, col3 = st.columns([1, 5, 1])
        with col1:
            st.write(f"{response['suraId']}, {response['verseID']}")

        with col2:

            st.write(response['indoText'])
        with col3:
            relevan = st.checkbox("", key=f"checkbox-{i}")

# submit ir
def add_feedback():
    supabase.table('feedback').insert({}).execute()
    pass

def main_program():
    pass
    
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


# feedbcak form
else :
    st.title("Hasil Pencarian")
    st.subheader(f"Oke, {st.session_state.username}!")
    st.markdown(f"### hasil pencarian: {st.session_state.query}")

    results = get_results(st.session_state.query)
    print_results(results)
    if st.button("Submit Feedback"):
        add_feedback()
        st.success("Feedback terkirim!")

        # reset ke halaman pencarian query
        st.session_state.query_is_submitted = False
        st.rerun()