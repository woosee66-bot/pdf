##### 기본 정보 입력 #####
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
import fitz

OPENAI_API = "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"

##### 기능 구현 함수 #####
@st.cache_resource
def make_retriever(uploaded_file):
    """업로드된 PDF로 Retriever 생성 (캐싱)"""
    pdf_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    docs = ""
    for page in doc:
        docs += page.get_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    split_documents = text_splitter.split_text(docs)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API)
    vectorstore = FAISS.from_texts(split_documents, embedding=embeddings)

    return vectorstore.as_retriever()


def make_response(retriever, question):
    """질문에 대한 답변 생성"""
    prompt = PromptTemplate.from_template("""
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Answer in Korean.

#Question:
{question}

#Context:
{context}

#Answer:
""")

    llm = init_chat_model(
        model="openai:gpt-5-nano",
        api_key=OPENAI_API,
        temperature=0
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)


##### 메인 함수 #####
def main():
    st.set_page_config(page_title="PDF Q&A", layout="wide")

    # 세션 상태 초기화
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.header("📄 PDF 내용 질문 프로그램")
    st.markdown("---")

    # PDF 업로드
    st.subheader("📤 PDF 파일을 업로드하세요")
    pdf = st.file_uploader(" ", type="pdf")

    if pdf is not None:
        with st.spinner("PDF 분석 중..."):
            retriever = make_retriever(pdf)

        st.success(f"✅ PDF 인덱싱 완료: {pdf.name}")
        st.markdown("---")

        # 질문 입력
        st.subheader("❓ 질문을 입력하세요")

        with st.form("question_form", clear_on_submit=True):
            user_question = st.text_input("PDF에 대해 무엇이든 물어보세요:")
            submitted = st.form_submit_button("질문하기")

        # 질문 제출 시 답변 생성 후 기록 저장
        if submitted and user_question:
            with st.spinner("답변 생성 중..."):
                response = make_response(retriever, user_question)

            st.session_state["chat_history"].append({
                "question": user_question,
                "answer": response
            })

        # 질문/답변 이력 출력
        if st.session_state["chat_history"]:
            st.markdown("---")
            st.subheader("💬 질문 및 답변 기록")

            for i, chat in enumerate(st.session_state["chat_history"], start=1):
                st.markdown(f"### Q{i}. {chat['question']}")
                st.info(chat["answer"])


if __name__ == "__main__":
    main()