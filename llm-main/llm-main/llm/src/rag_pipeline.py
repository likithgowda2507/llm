import os
import re
from typing import Dict, Any, List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests

class SOPRagPipeline:
    def __init__(
        self,
        pdf_dir: str,
        vector_db_path: str = "faiss_index",
        llm_provider: str = "",
        groq_model: str = "",
        groq_base_url: str = "",
    ):
        self.pdf_dir = pdf_dir
        self.vector_db_path = vector_db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "")
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "")
        self.groq_base_url = groq_base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        self.llm = None

    def _setup_llm(self):
        """Sets up an LLM. Defaults to Groq if GROQ_API_KEY is set, else local."""
        provider = (self.llm_provider or "").lower().strip()
        if provider == "local":
            return self._setup_local_llm()
        if provider == "groq" or self.groq_api_key:
            return self._setup_groq_llm()
        return self._setup_local_llm()

    def _setup_local_llm(self):
        """Sets up a local LLM using HuggingFace."""
        model_id = "MBZUAI/LaMini-T5-738M"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        def generate_text(prompt) -> str:
            if not isinstance(prompt, str):
                if hasattr(prompt, "to_string"):
                    prompt = prompt.to_string()
                else:
                    prompt = str(prompt)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return RunnableLambda(generate_text)

    def _setup_groq_llm(self):
        """Sets up Groq via OpenAI-compatible Chat Completions."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for Groq API usage.")
        if not self.groq_model:
            raise ValueError("GROQ_MODEL is required for Groq API usage.")

        base_url = self.groq_base_url.rstrip("/")
        endpoint = f"{base_url}/chat/completions"

        def generate_text(prompt) -> str:
            if not isinstance(prompt, str):
                if hasattr(prompt, "to_string"):
                    prompt = prompt.to_string()
                else:
                    prompt = str(prompt)

            payload = {
                "model": self.groq_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "max_tokens": 512,
            }
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
            if response.status_code >= 400:
                raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")

            data = response.json()
            return data["choices"][0]["message"]["content"]

        return RunnableLambda(generate_text)

    def load_and_process_documents(self):
        """Loads PDFs, chunks them, and creates/saves the vector store."""
        documents = []
        if not os.path.exists(self.pdf_dir):
            print(f"Error: Directory {self.pdf_dir} not found.")
            return

        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        
        print(f"Loading {len(pdf_files)} PDF documents...")
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(os.path.join(self.pdf_dir, pdf_file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = pdf_file
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")

        if not documents:
            print("No documents were loaded.")
            return

        # Text Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunks.")

        # Vector Store Creation
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.vector_db_path)
        print(f"Vector store saved to {self.vector_db_path}")

    def load_vector_store(self):
        """Loads an existing vector store."""
        if os.path.exists(self.vector_db_path):
            self.vector_store = FAISS.load_local(
                self.vector_db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print("Vector store loaded successfully.")
        else:
            print("Vector store not found. Please run indexing first.")

    def format_docs(self, docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            if page is not None:
                formatted.append(f"Source: {source} (page {page + 1})\n{doc.page_content}")
            else:
                formatted.append(f"Source: {source}\n{doc.page_content}")
        return "\n\n".join(formatted)

    def _normalize_text(self, text: str) -> str:
        lowered = (text or "").lower()
        cleaned = []
        for ch in lowered:
            if ch.isalnum() or ch.isspace():
                cleaned.append(ch)
            else:
                cleaned.append(" ")
        cleaned = "".join(cleaned)
        cleaned = cleaned.replace(" sop ", " ")
        cleaned = " ".join(cleaned.split())
        return cleaned

    def _extract_source_hint(self, question: str) -> Optional[str]:
        if not self.vector_store:
            return None

        question_norm = self._normalize_text(question)
        if not question_norm:
            return None

        sources = set()
        try:
            for doc_id in self.vector_store.index_to_docstore_id.values():
                doc = self.vector_store.docstore._dict.get(doc_id)
                if doc is None:
                    continue
                source = doc.metadata.get("source", "")
                if source:
                    sources.add(source)
        except Exception:
            return None

        best_source = None
        best_len = 0
        for source in sources:
            name = os.path.splitext(source)[0]
            name_norm = self._normalize_text(name)
            if not name_norm:
                continue
            if name_norm in question_norm and len(name_norm) > best_len:
                best_source = source
                best_len = len(name_norm)

        return best_source

    def _filter_docs_by_source(self, docs: List, source_hint: Optional[str]) -> List:
        if not source_hint:
            return docs
        hint_lower = source_hint.lower()
        filtered = [doc for doc in docs if hint_lower in doc.metadata.get("source", "").lower()]
        return filtered or docs

    def retrieve_docs(self, question: str, k: int = 6) -> List:
        if not self.vector_store:
            self.load_vector_store()
        if not self.vector_store:
            return []
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        source_hint = self._extract_source_hint(question)
        docs = retriever.invoke(question)
        return self._filter_docs_by_source(docs, source_hint)

    def _is_low_quality_answer(self, answer: str) -> bool:
        if not answer:
            return True
        trimmed = answer.strip()
        if len(trimmed) < 20:
            return True
        low = trimmed.lower()
        return low in {"yes.", "no.", "yes", "no", "i don't know", "idk"}



    def answer_question(self, question: str) -> Dict[str, Any]:
        """Retrieves context and generates an answer."""
        if not self.vector_store:
            self.load_vector_store()
        
        if not self.vector_store:
            return {"answer": "Vector store not initialized.", "sources": []}

        if self.llm is None:
            self.llm = self._setup_llm()

        template = """You are a strict QA assistant for SOP documents.
Use ONLY the context below. If the answer is not explicitly stated in the context,
say: "Not found in the provided SOPs." Do not infer or assume.
Do not answer with "Yes" or "No" alone.
If the question is too broad or ambiguous, respond with: "Not found in the provided SOPs."
Include all relevant details found in the context, including any policy, confidentiality, or NDA statements.

Context: {context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})
        source_hint = self._extract_source_hint(question)
        source_docs = retriever.invoke(question)
        source_docs = self._filter_docs_by_source(source_docs, source_hint)

        if not source_docs:
            return {"answer": "Not found in the provided SOPs.", "sources": []}

        context_text = self.format_docs(source_docs)

        rag_chain = (
            {"context": RunnableLambda(lambda _: context_text), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        sources = list(set([doc.metadata.get("source", "Unknown") for doc in source_docs]))
        answer = rag_chain.invoke(question)
        if self._is_low_quality_answer(answer):
            answer = "Not found in the provided SOPs."
        
        return {
            "answer": answer,
            "sources": sources
        }

    def extract_only(self, question: str, k: int = 6) -> Dict[str, Any]:
        """Returns verbatim text chunks from PDFs with source + page, no summarization."""
        if not self.vector_store:
            self.load_vector_store()

        if not self.vector_store:
            return {"answer": "Vector store not initialized.", "sources": [], "excerpts": []}

        docs = self.retrieve_docs(question, k=k)
        if not docs:
            return {"answer": "Not found in the provided SOPs.", "sources": [], "excerpts": []}

        excerpts = []
        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            header = f"{source} (page {page + 1})" if page is not None else source
            excerpts.append(f"{header}\n{doc.page_content}")
            sources.append(source)

        return {
            "answer": "",
            "sources": list(set(sources)),
            "excerpts": excerpts,
        }

    def summarize_question(self, question: str, k: int = 6) -> Dict[str, Any]:
        """Returns a concise summary of relevant SOP text with source + page references."""
        if not self.vector_store:
            self.load_vector_store()

        if not self.vector_store:
            return {"summary": "Vector store not initialized.", "sources": []}

        if self.llm is None:
            self.llm = self._setup_llm()

        docs = self.retrieve_docs(question, k=k)
        if not docs:
            return {"summary": "Not found in the provided SOPs.", "sources": []}

        context_text = self.format_docs(docs)

        template = """Summarize the relevant SOP content below in 2-4 concise sentences.
Use ONLY the context. Do not add or infer details not present.
If the context does not answer the question, respond with: "Not found in the provided SOPs."

Context: {context}

Question: {question}

Summary:"""

        prompt = PromptTemplate.from_template(template)
        summarizer = (
            {"context": RunnableLambda(lambda _: context_text), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        summary = summarizer.invoke(question)

        source_refs = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            source_refs.append(f"{source} (page {page + 1})" if page is not None else source)

        return {
            "summary": summary,
            "sources": sorted(set(source_refs)),
        }

    def generate_text(self, prompt: str) -> str:
        """Direct LLM call for auxiliary tasks (e.g., flowchart reconstruction)."""
        if self.llm is None:
            self.llm = self._setup_llm()
        if hasattr(self.llm, "invoke"):
            return self.llm.invoke(prompt)
        return self.llm(prompt)
