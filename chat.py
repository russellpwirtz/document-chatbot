from dotenv import load_dotenv
import os

load_dotenv()

# NOTE: copy .env.example to .env and update values!
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model_name = os.getenv("OPENAI_MODEL_NAME")
serp_api_key = os.getenv("SERPAPI_API_KEY")

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

# Imports
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_community.vectorstores import Chroma
import gradio as gr
import time

# Data Ingestion
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.xml import UnstructuredXMLLoader
from langchain.document_loaders.csv_loader import CSVLoader

loaders = {
    '.pdf': PyMuPDFLoader,
    '.xml': UnstructuredXMLLoader,
    '.csv': CSVLoader,
}

def create_directory_loader(file_type, directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=loaders[file_type],
    )

# Create DirectoryLoader instances for each file type
pdf_loader = create_directory_loader('.pdf', 'data/')
xml_loader = create_directory_loader('.xml', 'data/')
csv_loader = create_directory_loader('.csv', 'data/')

loaders = [pdf_loader, xml_loader, csv_loader]
documents = []
for loader in loaders:
    documents.extend(loader.load())

# Chunk and Embeddings
text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=1500)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents,
    embeddings,
    collection_name="my_documents",
    persist_directory=".chromadb/"
)

chat_history = []

general_system_template = r"""
Given a specific context and chat history, please give a concise and helpful response.
----
Context:
{context}
----
History:
{chat_history}
----
"""
general_user_template = "Question:```{question}```"
messages = [
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template(general_user_template),
]
qa_prompt = ChatPromptTemplate.from_messages(messages)


# Create the multipurpose chain
qachat = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.6, model_name=openai_model_name, max_tokens=200),
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    verbose=True,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
)


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        chat_tuple = [tuple(chat) for chat in chat_history]
        # print(f"Chat tuple: {str(chat_tuple)}")
        result = qachat({"question": message, "chat_history": chat_tuple})
        chat_history.append((message, result["answer"]))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(debug=True)
