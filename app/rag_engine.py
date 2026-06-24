from dotenv import load_dotenv
import os
import re
from pathlib import Path
from typing import List, Dict, Any

from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ==================================================
# Load Environment Variables
# ==================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# ==================================================
# Groq Client
# ==================================================
client = Groq(api_key=GROQ_API_KEY)

# ==================================================
# Embedding Model
# ==================================================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================================
# Load Chroma Database
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "vector_db"

if not DB_PATH.exists():
    print(f"Warning: vector_db folder not found at {DB_PATH}")

print("Loading DB from:", DB_PATH)

vector_db = Chroma(
    persist_directory=str(DB_PATH),
    embedding_function=embedding_model
)

# ==================================================
# Retriever
# ==================================================
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ==================================================
# Helpers
# ==================================================
def _normalize(text: str) -> str:
    return (text or "").strip().lower()

def _looks_like_pdf_reference(text: str) -> bool:
    text = (text or "").strip()
    return bool(re.search(r"\.pdf$", text, re.IGNORECASE))

def _dedupe_docs_by_file(docs):
    unique_docs = []
    seen_files = set()

    for doc in docs:
        file_name = doc.metadata.get("source_file", "Unknown File")
        key = _normalize(file_name)
        if key not in seen_files:
            unique_docs.append(doc)
            seen_files.add(key)

    return unique_docs

def _extract_pdf_names(text: str) -> List[str]:
    if not text:
        return []
    files = re.findall(r"\b[\w.-]+\.pdf\b", text, flags=re.IGNORECASE)
    return sorted(set(files))

def _build_context(docs, limit_per_doc: int = 1200) -> str:
    context = ""

    for doc in docs:
        contract_type = doc.metadata.get("contract_type", "Unknown Contract Type")
        source_file = doc.metadata.get("source_file", "Unknown File")
        page_number = doc.metadata.get("page", "Unknown Page")

        context += f"""
FILE NAME:
{source_file}

CONTRACT TYPE:
{contract_type}

PAGE:
{page_number}

CONTENT:
{doc.page_content[:limit_per_doc]}

--------------------------------------------------
"""
    return context

def _llm_call(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def _empty_response(question: str, no_match_text: str = "No matching contracts found.") -> Dict[str, Any]:
    return {
        "answer": f"""Answer:
{no_match_text}

Relevant Contract Files:
None

Evidence:
No supporting evidence was retrieved from the vector database.
""",
        "files": [],
        "retrieved_count": 0,
        "question": question
    }

def _exact_file_search(file_name: str, k: int = 50):
    """
    Exact metadata match on source_file.
    This is the key fix for the retrieval bug.
    """
    file_name = (file_name or "").strip()
    if not file_name:
        return []

    docs = []

    # Primary path: Chroma metadata filter
    try:
        docs = vector_db.similarity_search(
            query=file_name,
            k=k,
            filter={"source_file": file_name}
        )
    except Exception:
        docs = []

    if docs:
        return _dedupe_docs_by_file(docs)

    # Fallback path: direct collection lookup
    try:
        raw = vector_db._collection.get(where={"source_file": file_name})
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []

        for page_content, metadata in zip(documents, metadatas):
            docs.append(
                Document(
                    page_content=page_content or "",
                    metadata=metadata or {}
                )
            )
    except Exception:
        pass

    return sorted(
    docs,
    key=lambda d: int(d.metadata.get("page", 0))
)

def _semantic_search(query: str, k: int = 5):
    try:
        docs = retriever.invoke(query)
    except Exception:
        docs = []

    return _dedupe_docs_by_file(docs)

def _retrieve_docs_for_contract(contract_ref: str, k: int = 10):
    """
    If user gives a filename like emp_01.pdf, use exact metadata filter.
    Otherwise, fall back to semantic search using the keyword/query.
    """
    contract_ref = (contract_ref or "").strip()
    if not contract_ref:
        return []

    if _looks_like_pdf_reference(contract_ref):
        exact_docs = _exact_file_search(contract_ref, k=max(k, 50))
        if exact_docs:
            return exact_docs

        # If exact filename failed, do NOT return unrelated files blindly.
        # Return empty so the UI can tell the truth.
        return []

    # Keyword-based search
    return _semantic_search(contract_ref, k=k)

def _retrieve_docs_for_query(query: str, k: int = 5):
    return _semantic_search(query, k=k)

# ==================================================
# 1) Question Answering / Clause Search
# ==================================================
def analyze_risks(query: str) -> Dict[str, Any]:
    try:
        docs = _retrieve_docs_for_query(query, k=5)

        if not docs:
            return _empty_response(query)

        source_files = sorted(
            set(doc.metadata.get("source_file", "Unknown File") for doc in docs)
        )

        context = _build_context(docs, limit_per_doc=1200)

        prompt = f"""
You are an expert legal contract assistant.

Only list a contract if the contract explicitly contains
the clause or concept asked in the question.

Do NOT list a contract merely because it was retrieved.

Ignore retrieved contracts that do not directly answer the question.

STRICT RULES:

1. NEVER use outside knowledge.
2. NEVER invent contract names.
3. ONLY mention contracts found in the context.
4. If evidence does not exist, say:
   No matching contracts found.
5. Keep answers concise and factual.
6. Quote short evidence when available.
7. Do not speculate.

Return EXACTLY in this format:

Answer:
<direct answer>

Relevant Contract Files:
- file1.pdf
- file2.pdf

Evidence:
<brief evidence from retrieved contracts>

CONTRACT CONTEXT:

{context}

QUESTION:

{query}
"""

        answer = _llm_call(prompt)
        matched_files = _extract_pdf_names(answer)

        if not matched_files:
            matched_files = source_files

        return {
            "answer": answer,
            "files": matched_files,
            "retrieved_count": len(matched_files),
            "question": query
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "files": [],
            "retrieved_count": 0,
            "question": query
        }

# ==================================================
# 2) Single Contract Summary + Risk Profile
# ==================================================
def summarize_contract(contract_ref: str) -> Dict[str, Any]:
    try:
        docs = _retrieve_docs_for_contract(contract_ref, k=10)

        if not docs:
            return {
                "answer": f"""Answer:
No matching contracts found for: {contract_ref}

Risk Categories:
- None

Overall Risk Score:
0

Overall Risk Level:
Low

Evidence:
No supporting evidence was retrieved from the vector database.

Contract File:
{contract_ref} (not found)
""",
                "files": [],
                "retrieved_count": 0,
                "question": contract_ref
            }

        source_files = sorted(
            set(doc.metadata.get("source_file", "Unknown File") for doc in docs)
        )

        context = _build_context(docs, limit_per_doc=1500)

        prompt = f"""
You are an expert legal contract analyst.

Analyze ONLY the contract context below.
Do not use outside knowledge.
Do not invent clauses.
If the contract is not clear, say so plainly.

Return EXACTLY in this format:

Answer:
<short contract summary>

Risk Categories:
- <category 1>
- <category 2>
- <category 3>

Overall Risk Score:
<number out of 100>

Overall Risk Level:
Low / Medium / High

Evidence:
<brief evidence from the contract>

Contract File:
<file name or file names>

CONTRACT CONTEXT:

{context}

QUESTION:

Give a summary and risk profile for this contract:
{contract_ref}
"""

        answer = _llm_call(prompt)
        matched_files = _extract_pdf_names(answer)

        if not matched_files:
            matched_files = source_files

        return {
            "answer": answer,
            "files": matched_files,
            "retrieved_count": len(matched_files),
            "question": contract_ref
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "files": [],
            "retrieved_count": 0,
            "question": contract_ref
        }

# ==================================================
# 3) Compare Two Contracts
# ==================================================
def compare_contracts(contract_a: str, contract_b: str) -> Dict[str, Any]:
    try:
        docs_a = _retrieve_docs_for_contract(contract_a, k=10)
        docs_b = _retrieve_docs_for_contract(contract_b, k=10)

        if not docs_a or not docs_b:
            missing = []
            if not docs_a:
                missing.append(contract_a)
            if not docs_b:
                missing.append(contract_b)

            return {
                "answer": f"""Answer:
No matching contracts found for comparison.

Relevant Contract Files:
None

Evidence:
Could not find exact contract data for: {", ".join(missing)}.
""",
                "files": [],
                "retrieved_count": 0,
                "question": f"{contract_a} vs {contract_b}"
            }

        context_a = _build_context(docs_a, limit_per_doc=1200)
        context_b = _build_context(docs_b, limit_per_doc=1200)

        files = sorted(
            set(
                [doc.metadata.get("source_file", "Unknown File") for doc in (docs_a + docs_b)]
            )
        )

        prompt = f"""
You are an expert legal contract analyst.

Compare the two contract contexts below.
Use ONLY the retrieved contract context.
Do not invent anything.
Do not use outside knowledge.

Return EXACTLY in this format:

Answer:
<short comparison summary>

Similarities:
- ...

Differences:
- ...

Key Risks:
- ...

Relevant Contract Files:
- file1.pdf
- file2.pdf

Evidence:
<brief evidence for both contracts>

CONTRACT A CONTEXT:
{context_a}

CONTRACT B CONTEXT:
{context_b}

QUESTION:
Compare these two contracts:
A = {contract_a}
B = {contract_b}
"""

        answer = _llm_call(prompt)
        matched_files = _extract_pdf_names(answer)

        if not matched_files:
            matched_files = files

        return {
            "answer": answer,
            "files": matched_files,
            "retrieved_count": len(matched_files),
            "question": f"{contract_a} vs {contract_b}"
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "files": [],
            "retrieved_count": 0,
            "question": f"{contract_a} vs {contract_b}"
        }