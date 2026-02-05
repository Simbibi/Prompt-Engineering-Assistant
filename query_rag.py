from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from api import OPENAI_API_KEY, OPENAI_API_BASE


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
)

vector_store = Chroma(
    collection_name="prompt_engineering",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

prompt = ChatPromptTemplate.from_template(
"""You are helpful assistant that can aswer questions about the blog post on promp engineering. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say 'I don't know.'
Question: {question}
Context: {context}
Answer:""")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
)

question = 'What is  Ferrrari?'

retrieved_docs = vector_store.similarity_search(question, k = 3)

docs_content = "\n".join([doc.page_content for doc in retrieved_docs])

message = prompt.invoke({'question': question, 'context': docs_content})

answer = llm.invoke(message)

print(answer.content)