import pandas as pd
from io import StringIO
import logging
import re
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import GROQ_API_KEY, MODEL_NAME

logger = logging.getLogger(__name__)

# Load dataset
df = pd.read_csv("data/remediations.csv")

df["combined"] = df.apply(lambda r: f"""
SHRP_ID: {r['shrp_id']}
Issue: {r['issue_summary']}
Root Cause: {r['root_cause']}
Timeline: {r['timeline']}
Impacted Customers: {r['impacted_customers']}
Resolution: {r['resolution']}
""", axis=1)

# Create docs
docs = DataFrameLoader(df, page_content_column="combined").load()

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
docs = splitter.split_documents(docs)

# USe if open AI is leveraged
# embeddings = OpenAIEmbeddings()  

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Create Prompt
template="""
You are a remediation analyst.

New Issue:
{question}

Context:
{context}

Return CSV only:

SHRP_ID,Issue_Summary,Root_Cause,Timeline,Impacted_Customers_Count,Resolution_Summary
"""

prompt = ChatPromptTemplate.from_template(template)


# Create Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME,
    temperature=0
)


# Create the Chain to format and clean the output
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_rag(issue: str):
    logger.info(f"Running RAG with Groq for: {issue}")
    
    # Ensure format_docs handles a list of Documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        # Use itemgetter or a lambda to ensure the string 'issue' 
        # is passed correctly to the retriever
        {"context": (lambda x: x["question"]) | retriever | format_docs, 
         "question": lambda x: x["question"]}

        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Pass as a dictionary to match the Runnable mapping above
    result = rag_chain.invoke({"question": issue})
    
    # Clean the output: Remove markdown code blocks if the LLM adds them
    clean_csv = re.sub(r'^```csv\s*|```\s*$', '', result.strip(), flags=re.MULTILINE)
    
    # Use StringIO to load into DataFrame
    try:
        return pd.read_csv(StringIO(clean_csv))
    except Exception as e:
        logger.error(f"Failed to parse CSV: {result}")
        # Fallback: Return a single-row DF with the error if parsing fails
        return pd.DataFrame([{"Error": "Could not parse LLM output", "Raw": result}])


