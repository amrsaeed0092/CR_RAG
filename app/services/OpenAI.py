import pandas as pd
from io import StringIO
import logging

from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from config import OPENAI_API_KEY, MODEL_NAME

# Setup logging
logger = logging.getLogger(__name__)

# Load dataset
df = pd.read_csv("remediations.csv")

df["combined"] = df.apply(lambda r: f"""
SHRP_ID: {r['shrp_id']}
Issue: {r['issue_summary']}
Root Cause: {r['root_cause']}
Timeline: {r['timeline']}
Impacted Customers: {r['impacted_customers']}
Resolution: {r['resolution']}
""", axis=1)

# Create documents
docs = DataFrameLoader(df, page_content_column="combined").load()

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(docs)

# Embeddings + Vector DB
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Prompt
prompt = PromptTemplate(
    template="""
You are a remediation analyst.

New Issue:
{question}

Context:
{context}

Return CSV only:

SHRP_ID,Issue_Summary,Root_Cause,Timeline,Impacted_Customers_Count,Resolution_Summary
""",
    input_variables=["context", "question"]
)

# LLM
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

def run_rag(issue: str) -> pd.DataFrame:
    logger.info(f"Running RAG for issue: {issue}")

    result = qa_chain.run(issue)

    logger.info("RAG completed successfully")

    return pd.read_csv(StringIO(result))