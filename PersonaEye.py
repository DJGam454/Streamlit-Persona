import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Caching the PDF loading and processing
@st.cache_resource
def load_and_process_pdf(file_path):
    loader = PyPDFLoader(file_path=file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    return chunks

# Caching the embedding model
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Caching the vector store creation
# Modified create_vector_store function
@st.cache_resource
def create_vector_store(_chunks, _embeddings):
    return FAISS.from_documents(_chunks, _embeddings)

# Caching the LLM
@st.cache_resource
def get_llm():
    return ChatOllama(model="llama3.1")

# Caching the retriever creation
@st.cache_resource
def create_retriever(_vector_store, _llm):
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to provide information 
        from the 'Personal Brand Workbook' to help the user enhance their personal brand and 
        effectively sell themselves. Based on the user's question, you should retrieve relevant 
        sections that focus on building a unique personal brand, highlighting strengths, 
        and improving self-presentation.

        If the user's question is about something unrelated to personal branding or selling 
        themselves, do not provide an answer and instead inform the user that the query is 
        outside the scope of this document.

        Your goal is to provide actionable insights that the user can apply to establish a 
        compelling personal brand. Ensure the alternative questions guide the user towards 
        discovering practical steps for personal branding.

        Original question: {question}"""
    )
    
    return MultiQueryRetriever.from_llm(
        _vector_store.as_retriever(), 
        _llm,
        prompt=QUERY_PROMPT,
    )

# Caching the chain creation
@st.cache_resource
def create_chain(_retriever, _llm):
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    return (
        {"context": _retriever, "question": RunnablePassthrough()}
        | prompt
        | _llm
        | StrOutputParser()
    )

# Main Streamlit app
def main():
    st.title('Ask PersonaEye 𓁼')

    # Load and process PDF
    local_path = "./personal_brand_workbook.pdf"
    chunks = load_and_process_pdf(local_path)

    # Get embeddings
    embeddings = get_embeddings()

    # Create vector store
    vector_store = create_vector_store(chunks, embeddings)

    # Get LLM
    llm = get_llm()

    # Create retriever
    retriever = create_retriever(vector_store, llm)

    # Create chain
    chain = create_chain(retriever, llm)

    # Setup a session state message variable to hold all the old messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display all the historical messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Get user input
    user_question = st.chat_input("Please enter your prompt:")

    # If the user hits enter
    if user_question:
        # Display the prompt
        st.chat_message('user').markdown(user_question)
        # Store the user prompt in state
        st.session_state.messages.append({'role':'user', 'content':user_question})
        
        try:
            # Send the prompt to the PDF Q&A chain
            response = chain.invoke(user_question)
            # Show the LLM response
            st.chat_message('assistant').markdown(response)
            # Store the LLM Response in state
            st.session_state.messages.append({'role': 'assistant', 'content':response})
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()