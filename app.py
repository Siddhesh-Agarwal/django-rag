from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from pydantic.v1 import SecretStr
import streamlit as st


@st.cache_resource
def get_retriever(api_key: str):
    """A Chroma retriever that uses OpenAI embeddings"""

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=SecretStr(api_key),
    )
    return Chroma(
        collection_name="django",
        embedding_function=embeddings,
        persist_directory="./chroma/",
    ).as_retriever()


query = st.text_input("Question")
openai_api_key = st.text_input("OpenAI API Key")
model = OpenAI(temperature=0)
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

if st.button("Submit"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key.")
    elif not query:
        st.error("Please enter a question.")
    else:
        retriever = get_retriever(openai_api_key)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        st.write_stream(chain.stream(query))
