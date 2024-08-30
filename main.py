import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

load_dotenv()

if __name__ == "__main__":
    print("Carga del archivo PDF...")
    pdf_path = os.getenv("FILE_PATH_PDF")
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    print("Dividir el PDF en chunks")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )

    docs = text_splitter.split_documents(documents=documents)

    print("Guardar los chunks en una base vectorial local")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_index_react")

    print("Carga de embeddings de la base vectorial local")
    new_vector_store = FAISS.load_local(
        folder_path="faiss_index_react", embeddings=embeddings, allow_dangerous_deserialization=True,
    )

    print("Creación y ejecución del agente")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )

    retrieval_chain = create_retrieval_chain(
        new_vector_store.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})

    print(res["answer"])

    print("Final del proceso")
