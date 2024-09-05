import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
import tempfile
import os
import requests

def is_valid_api_key(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(
            "https://api.openai.com/v1/models", headers=headers)
        return response.status_code == 200
    except requests.RequestException:
        return False

st.title("📄 Document GPT")
st.markdown(
    """
## Welcome!

챗봇을 사용하여 AI에게 파일에 대해 질문하세요!

openAPI 사용을 위해 sidebar에 API Key를 입력후 문서를 업로드 해주세요.

API 키는 저장되지 않습니다.
"""
)

cache_dir = LocalFileStore("./.cache/")
uploaded_file = ""

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("OpenAI API 키를 입력해주세요.", icon="🔑")
    if openai_api_key:
        if is_valid_api_key(openai_api_key):
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("API 키가 유효합니다.")
            uploaded_file = st.file_uploader(
        "문서 파일을 업로드해주세요 (.txt, .pdf, .docx)", type=("txt", "pdf", "docx")
    )
        else:
            st.warning("유효한 API 키를 입력하세요.")



question = st.text_area(
    "문서에 대해 질문을 하세요",
    placeholder="문서 요약좀 해줄래?",
    disabled=not uploaded_file,
)

if openai_api_key and uploaded_file and question:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.1)

    # 파일 형식에 맞는 로더 선택
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    if file_extension == ".pdf":
        loader = PyPDFLoader(tmp_file_path)  # PDF 파일 처리
    elif file_extension == ".txt":
        loader = TextLoader(tmp_file_path)  # 텍스트 파일 처리
    elif file_extension == ".docx":
        loader = UnstructuredFileLoader(tmp_file_path)  # DOCX 파일 처리
    else:
        st.error("지원되지 않는 파일 형식입니다.")
        st.stop()

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 주어진 문서를 읽고 해석합니다. 주어진 문서를 사용하여 질문에 상세하게 대답하십시오. 답을 모르면 지어내지 말고 모른다고 대답하십시오.\n\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def chain_with_memory(question):
        chat_history = memory.load_memory_variables({}).get("chat_history", "")

        relevant_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        inputs = {
            "context": context,
            "question": question,
            "chat_history": chat_history,
        }

        prompt = prompt_template.format_messages(**inputs)
        response = llm(prompt)

        memory.save_context({"question": question}, {"answer": response.content})

        return response.content

    if st.button("질문에 답변 받기"):
        with st.spinner("GPT가 문서를 읽고 답변 중입니다..."):
            answer = chain_with_memory(question)
            st.write("### 🤖")
            st.write(answer)

# 오류 또는 알림 처리
elif not openai_api_key:
    st.info("OpenAI API 키를 입력해주세요.")
elif not is_valid_api_key(openai_api_key):
    st.warning("유효한 API 키를 입력하세요.")
elif not uploaded_file:
    st.info("문서를 업로드해주세요.")
elif not question:
    st.info("문서에 대해 질문을 입력해주세요.")
