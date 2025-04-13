import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv


load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Q&A Chatbot with GROQ"

system_prompt = "You are a helpful assistant. Please respond to the user queries."
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user","Question: {question}")
    ]
)
def create_llm(name,groq_api_key):
    return ChatGroq(api_key=groq_api_key,model=name)

contextualize_q_system_prompt = ( #prompt to include chat history
    """Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is.
"""
)

history_aware_q_prompt = ChatPromptTemplate.from_messages([
    ("system",contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("user","{input}"),
])


st.title("Streamlit Q&A Chatbot with GROQ")
st.write("Enter your question below:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content=system_prompt)]
    #st.session_state.chat_history.append(f"Human Message: {SystemMessage(content=system_prompt)}")

user_input = st.text_input("Your Question:")
llm_name = st.sidebar.selectbox("Select Grog Model:",["Llama3-8b-8192","Gemma2-9b-It","Llama3.1-8b-Instant"])
api_key = st.sidebar.text_input("Enter your Groq API key",type="password")

temperature = st.sidebar.slider("Tempurature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

if user_input and api_key:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    llm = create_llm(llm_name,api_key)
    history_aware_retriever = history_aware_q_prompt|llm|StrOutputParser()
    history_aware_question = history_aware_retriever.invoke({
        "input":user_input,
        "chat_history":st.session_state.chat_history
    })
    qa_chain = qa_prompt|llm|StrOutputParser()
    response = qa_chain.invoke({
        "question":history_aware_question,
        "chat_history":st.session_state.chat_history
    })
    st.session_state.chat_history.append(AIMessage(content=response))
    st.write(response)
