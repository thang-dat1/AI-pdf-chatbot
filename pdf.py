
# upload PDF 
# --> Doc text tu PDF
# --> Cat text thanh chunk nho
# --> Bien chunk thanh vector embedding
# --> Luu vector vao FAISS 
# --> user hoi 
# --> Bien question thanh vector
# --> FAISS tim chunk gan nghia nhat
# --> Dua chunk vao prompt
# --> LLM doc context 
# --> LLM tra loi 

# Large Language Model la co may du doan tu tiep theo, VD: toi dang an... LLM doan: com, chicken, bimbim...
# no hoc hang ti cau tren internet nen du doan rat ao ma

# IMPORT THU VIEN --------------------
import streamlit as st     
# tao web app / chat UI / upload file / spinner / session memory 
# as : viet tat streamlit = st
                                           
from langchain_community.document_loaders import PyPDFLoader
# mo file / extract text / chia theo page

from langchain_text_splitters import RecursiveCharacterTextSplitter
# cat text thanh chunk nho cho AI de nuot

from langchain_community.vectorstores import FAISS
# FAISS: vector database chuyen dung de luu vector --> semantic(nghia)/similarity(giong) search (tim ND gan nghia nhat)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# Neu dung Ollama local --> dung OllamaEmbeddings va Ollama
# =========================
# from langchain_community.embeddings import OllamaEmbeddings
# bien text thanh vector (VD "con meo" = [0.123, -0.567, ...])
# AI se dung vector de hieu y nghia
# from langchain_community.llms import Ollama 
# LLM chay local bang Ollama, ket noi Python voi model AI local

import tempfile
# tao file tam
# Vi Streamlit upload chua phai file that tren o cung

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="centered"  # UI nam giua man hinh.
)

st.title("📄 AI PDF Chatbot")

# =========================
# SIMPLE CHAT DETECTOR
# =========================
def is_small_talk(text):             # function ktra dang chat binh thuong hay hoi ve PDF

    text = text.lower().strip()      # strip, xoa khoang trang hoac ki tu thua o dau&cuoi chuoi

    single_words = [
        "hi",
        "hello",
        "chao",
        "thank",
        "thanks",
        "good",
        "ok",
        "oke",
        "haha",
        "bye",
        "yo",
        "helo",
        "hey"
    ]

    phrases = [
        "xin chao",
        "cam on",
        "tam biet"
    ]

    words = text.split()

    return (
        any(word in words for word in single_words)
        or any(phrase in text for phrase in phrases)    
    )


# =========================
# CREATE VECTOR DATABASE
# =========================
@st.cache_resource
# (!) no bao Streamlit: Function nay chay 1 lan thoi, dung build lai
# neu khong co cache: moi lan chat --> vector DB se build lai

def create_db(pdf_path):             # function tao vector database  

    loader = PyPDFLoader(pdf_path)   # 2 dong nay doc PDF (ext, metadata, page number)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,              # moi chunk 500 kitu
        chunk_overlap=50             # overlap 50 kitu (moi doan trung nhau 50 kitu, tranh mat context)
    )

    docs = splitter.split_documents(documents)  # PDF lon --> nhieu doan (chunk) nho
    
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # embeddings = OllamaEmbeddings(              # retrieval: truy xuat, tim thong tin lien quan nhat
    #     model="bge-m3"
    # )

    db = FAISS.from_documents(                  # text --> embedding --> vector --> FAISS luu
        docs,
        embeddings
    )

    return db                                   # tra ve vector database 


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_llm():                                 # function load model AI (LLM: Large Language Model)
    return ChatGoogleGenerativeAI(
        model="gemini-1.0-pro",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.3
    )
    # return Ollama(
    #     model="qwen2.5:7b"                      # load Qwen model
    # )


# =========================
# UPLOAD PDF
# =========================
uploaded_file = st.file_uploader(               # Nut UPLOAD file
    "Upload PDF",
    type="pdf"
)

if uploaded_file:                               # ktra xem co file chua, co file upload roi moi chay

    # =========================
    # SAVE TEMP FILE
    # =========================
    with tempfile.NamedTemporaryFile(           # tao file tam tren may 
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(
            uploaded_file.read()                # ghi PDF vao o cung
        )

        pdf_path = tmp_file.name                # lay duong dan file 

    # =========================
    # LOAD DATABASE
    # =========================
    with st.spinner("📚 Đang đọc PDF..."):

        db = create_db(pdf_path)               # doc PDF, split, embedding, vector DB

    # =========================
    # LOAD LLM
    # =========================
    llm = load_llm()                          

    # =========================
    # CHAT MEMORY
    # =========================
    if "messages" not in st.session_state:     # tao memory chat
        st.session_state.messages = []         # st session_state: RAM cua chat bot
                                               # luu history, setting, trang thai
    # =========================
    # SHOW OLD CHAT
    # =========================
    for message in st.session_state.messages:  # render chat cu

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # =========================
    # CHAT INPUT
    # =========================
    question = st.chat_input(                  # o chat kieu Chat GPT
        "Hỏi gì đó về file PDF..."
    )

    # =========================
    # USER ASKED QUESTION
    # =========================
    if question:

        # =========================
        # SAVE USER MESSAGE
        # =========================
        st.session_state.messages.append({     # luu lich su chat
            "role": "user",
            "content": question
        })

        # =========================
        # SHOW USER MESSAGE
        # =========================
        with st.chat_message("user"):           # Hien thi Bubble USER (hien thi o chat cua nguoi dung tren giao dien chatbot)

            st.markdown(question)

        # =========================
        # AI RESPONSE
        # =========================
        with st.chat_message("assistant"):     # Hien thi Bubble AI

            message_placeholder = st.empty()   # Tao box rong (dung de TYPING EFFECT, STREAMING)

            # =========================
            # SMALL TALK MODE
            # =========================
            if is_small_talk(question):        # Neu la chat thuong se khong search PDF
                
                # Prompt cho casual chat
                chat_prompt = f"""             
                Người dùng đang trò chuyện bình thường.

                Hãy trả lời thân thiện,
                tự nhiên,
                ngắn gọn bằng cả tiếng Anh và tiếng Việt.

                User: {question}
                """

                full_response = ""

                for chunk in llm.stream(chat_prompt):   # STREAMING RESPONSE 
                    # AI phan hoi ra tung token nho, giong Chat GPT dang go chu 
                    full_response += chunk.content

                    message_placeholder.markdown(
                        full_response + "▌"            # tao hieu ung typing cursor (typing effect)
                    )

                message_placeholder.markdown(
                    full_response
                )

                response = full_response

            # =========================
            # PDF QA MODE
            # =========================
            else:               
                # neu khong phai chat casual --> dung RAG: Retrieval-Augmented Generation 
                # RAG la mot ky thuat/ kien truc/ phuong phap ket hop: tim du lieu dua vao prompt de LLM tra loi
                # Fine-tunning = day lai brain AI, RAG: dua AI tai lieu luc can

                message_placeholder.markdown(
                    "🤖 Đang tìm thông tin trong PDF..."
                )

                # =========================
                # SEARCH RELEVANT DOCS
                # =========================
                matched_docs = db.similarity_search(     # Semantic search: Question --> embedding --> tim vector gan nhat
                    question,
                    k=6
                )

                # =========================
                # CREATE CONTEXT
                # =========================
                context = "\n\n".join(                   # ghep chunk lai thanh context
                    [
                        doc.page_content
                        for doc in matched_docs
                    ]
                )

                # =========================
                # DEBUG CONTEXT
                # =========================
                with st.expander("🔍 Xem context AI đọc"):    # Cho xem AI dang doc gi

                    st.write(context)

                # =========================
                # CREATE PROMPT
                # =========================
                
                # Main RAG Prompt (Prompt RAG chinh)
                prompt = f"""
                Bạn là AI trợ lý đọc tài liệu PDF.

                Chỉ được trả lời dựa trên nội dung
                trong context.

                Nếu không tìm thấy thông tin thì trả lời:

                "Không tìm thấy thông tin trong tài liệu."

                Không được tự bịa thông tin.

                Luôn trả lời bằng cả tiếng Anh và tiếng Việt.

                =====================
                CONTEXT:
                =====================

                {context}

                =====================
                QUESTION:
                =====================

                {question}
                """

                # =========================
                # STREAM RESPONSE
                # =========================
                full_response = ""

                for chunk in llm.stream(prompt):      # AI generate từng token. (tao ra tung manh nho)

                    full_response += chunk.content

                    message_placeholder.markdown(
                        full_response + "▌"
                    )

                # remove cursor
                message_placeholder.markdown(
                    full_response
                )

                response = full_response             # luu ket qua cuoi

        # =========================
        # SAVE AI MESSAGE
        # =========================
        st.session_state.messages.append({           # luu tin nhan AI vao memory
            # append: them 1 phan tu vao cuoi danh sach 
            # extend, insert, remove, pop, clear, sort, reverse, count, index

            "role": "assistant",
            "content": response
        })

        # TONG KET: DAY LA KIEN TRUC RAG CHATBOT
        # PDF --> Chunking --> Embedding --> FAISS --> Question --> Similarity Search --> Context --> LLM --> Answer