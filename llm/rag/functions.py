from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import FAISS

import faiss
from dotenv import load_dotenv
import os
import time
import tempfile
import shutil
import requests
import os

# Prompt for the chatbot
prompt_template = PromptTemplate(
    input_variables=['context','question'],
    template="""
    You are a helpful AI assistant.
    If anyone is asking about your name, remember that your name is Jungler
    Be clear and complete to the answer. You may rephrase the answer to be more creative.

    Be warm and polite,Be precise and complete to the answer, and more creative

    Use the following context to answer the question about Junge Leiter.
    Context: {context}

    Question: {question}
    if the {question} is about introductions or greetings or small talks, you answer them politely and creatively, and give the creative follow up question
    if the {question} is about something that out of context of junge leiter, you may answer in dynamic way sentence-like "Let's focus on our discussion about Junge Leiter"
    If the {question} is about junge leiter, and you don't know the answer, just answer with sentence like "I don't have this information, you may contact Junge Leiter Team ..."
    Answer:
    """
)

# Prompt for generating standalone questions from the chat history
question_gen_prompt = PromptTemplate.from_template("""
  Given the following conversation and a follow-up question,
  Chat History:
  {chat_history}
  Follow-up Input: {question}
  Standalone question:
""")

_ = load_dotenv()
# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DOCUMENT_PATH = "https://drive.google.com/uc?id=1p7mazum0Z0lcad4krZNH2X50AGBEPhAY&export=download"

def load_document():
  # Download the PDF using requests
  response = requests.get(DOCUMENT_PATH, stream=True)  # Enable streaming
  response.raise_for_status()  # Raise an exception for bad responses

  # Save temporary PDF file
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
    with response.raw as raw_response:
        shutil.copyfileobj(raw_response, temp_pdf)
    temp_pdf_path = temp_pdf.name

  # Load the PDF file from temporary path
  loader = PyPDFLoader(temp_pdf_path)
  documents = loader.load()

  return documents

def load_preprocessed_document():
  # Load the PDF document
  documents = load_document()

  # split document instance
  doc_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=200,
)

  # Split the documnet
  docs = doc_splitter.split_documents(documents)
  return docs


def load_embeddings_db():
  # Load the processed document
  docs = load_preprocessed_document()
  
  # initiate the embedding model
  embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
  )

  # Store Vector in db
  db_vector = FAISS.from_documents(docs, embeddings)
  
  return db_vector

def get_retriever_qa():
  # load vector db
  db_vector = load_embeddings_db()

  # set base chat model LLM
  base_model = "gemini-2.0-flash"
  llm_model = ChatGoogleGenerativeAI(
    model=base_model,
    api_key=GOOGLE_API_KEY,
    temperature=0.3,
  )

  # Set the vector retriever
  retriever = db_vector.as_retriever()

  # Question generator chain
  question_generator = LLMChain(
    llm=llm_model,
    prompt=question_gen_prompt
  )

  # Chain to combine context and generate answers
  combine_docs_chain = load_qa_chain(
    llm=llm_model,
    chain_type="stuff",
    prompt=prompt_template
  )

  # Create memory
  memory = ConversationBufferMemory(
    memory_key="chat_history",  # Must match key used in chain
    return_messages=True,
    output_key="answer"
  )

  # Full ConversationalRetrievalChain
  qa_retriever = ConversationalRetrievalChain(
      retriever=retriever,
      question_generator=question_generator,
      combine_docs_chain=combine_docs_chain,
      memory=memory,
      return_source_documents=True,
  )
  return qa_retriever


chat_history = []  # Used to show message bubbles
def respond(user_input):
    global chat_history
    qa_retriever = get_retriever_qa()

    # Run the chatbot pipeline
    response = qa_retriever.invoke({
        'question': user_input,
        'chat_history': chat_history
    })

    # chunk response for streaming
    response_chunk = response['answer'].split(" ")

    partial_response = ""
    # Update history for the UI with output streams
    chat_history.append((user_input, ""))

    for char in response_chunk:
      time.sleep(0.01)
      partial_response += char + " "
      chat_history[-1] = (user_input, partial_response)
      yield chat_history, ""


def reset_memory():
    global chat_history
    chat_history = []

    return [], "Memory has been reset!"