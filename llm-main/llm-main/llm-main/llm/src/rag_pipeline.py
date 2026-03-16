import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

# Load .env from the project root (one level up from src/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

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
                "max_tokens": 1024,
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

    def _match_pdf_file(self, question: str) -> Optional[str]:
        """Match a PDF filename from the user's question using fuzzy name matching."""
        question_norm = self._normalize_text(question)
        if not question_norm:
            return None

        pdf_dir = Path(self.pdf_dir)
        if not pdf_dir.exists():
            return None

        best_match = None
        best_len = 0
        for pdf in pdf_dir.glob("*.pdf"):
            name_norm = self._normalize_text(pdf.stem)
            if not name_norm:
                continue
            # Check if the normalized PDF name appears in the question
            if name_norm in question_norm and len(name_norm) > best_len:
                best_match = pdf.name
                best_len = len(name_norm)

            # Also check individual significant words from the PDF name
            name_words = set(name_norm.split())
            # Remove very common/short words
            name_words -= {"sop", "ut", "of", "and", "the", "in", "for", "a", "an", "to"}
            question_words = set(question_norm.split())
            overlap = name_words & question_words
            # If most significant words match, it's the right PDF
            if len(name_words) > 0 and len(overlap) >= max(2, len(name_words) * 0.5):
                score = len(overlap)
                if score > best_len:
                    best_match = pdf.name
                    best_len = score

        return best_match

    def _get_targeted_pdf_text(self, pdf_filename: str, question: str = "", max_chars: int = 20000) -> str:
        """Read text content of a PDF file. Selects most relevant pages if it exceeds max_chars."""
        from pypdf import PdfReader
        import re

        pdf_path = Path(self.pdf_dir) / pdf_filename
        if not pdf_path.exists():
            return ""

        try:
            reader = PdfReader(str(pdf_path))
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((i, text))
            
            # If total length is within limits, return everything
            total_len = sum(len(text) for _, text in pages)
            if total_len <= max_chars:
                return "\n\n".join([f"--- {pdf_filename} (page {i + 1}) ---\n{text}" for i, text in pages])

            # Normalize query words, removing common stop words
            q_words = set(re.findall(r'\w+', question.lower())) - {"sop", "the", "a", "an", "of", "and", "in", "to", "for", "from", "flow", "chart", "table", "process", "what", "is", "diagram"}
            
            # Score each page
            page_scores = []
            for idx, (i, text) in enumerate(pages):
                text_lower = text.lower()
                # 1 point per matching keyword
                score = sum(3 for w in q_words if w in text_lower)
                # Boost if it contains exactly "process description" (crucial for flowcharts)
                if "process description" in text_lower:
                    score += 15
                page_scores.append((score, idx, i, text))
            
            # Sort by score descending
            page_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Select top pages until we hit max_chars.
            selected_indices = set()
            current_chars = 0
            
            for score, idx, i, text in page_scores:
                # Format page text
                page_text = f"--- {pdf_filename} (page {i + 1}) ---\n{text}"
                if current_chars + len(page_text) > max_chars and selected_indices:
                    break # Stop if we can't fit this page and we already have some
                
                selected_indices.add(idx)
                current_chars += len(page_text)
                
                # Try to add the next adjoining page if it fits (context spill over)
                if idx + 1 < len(pages) and idx + 1 not in selected_indices:
                    next_text = pages[idx+1][1]
                    next_page_text = f"--- {pdf_filename} (page {pages[idx+1][0] + 1}) ---\n{next_text}"
                    if current_chars + len(next_page_text) <= max_chars:
                        selected_indices.add(idx + 1)
                        current_chars += len(next_page_text)
            
            # Re-sort selected pages by their original order (idx)
            selected_pages = [pages[idx] for idx in sorted(list(selected_indices))]
            return "\n\n".join([f"--- {pdf_filename} (page {i + 1}) ---\n{text}" for i, text in selected_pages])
        except Exception:
            return ""

    def retrieve_docs(self, question: str, k: int = 10) -> List:
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
        if self.llm is None:
            self.llm = self._setup_llm()

        template = """You are an expert Quality SOP assistant. Answer questions accurately using ONLY the context below.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
1. First, give a clear and direct ANSWER to the question. Be thorough and detailed.
2. Then add a blank line and a "---" separator.
3. Then add a "**References:**" section listing each source with:
   - Document name and page number
   - The exact relevant lines quoted from that page (use quotation marks)

Example format:
[Your detailed answer here]

---
**References:**
- **SOP-Example Document.pdf (page 5):** "exact line from the document that supports the answer"
- **SOP-Another Document.pdf (page 3):** "another exact supporting line"

Rules:
- Use ONLY the context. Do NOT make up information.
- If the answer is not found, respond with: "Not found in the provided SOPs."
- Quote the EXACT lines from the context as references. Do not paraphrase.
- Include ALL relevant source pages, not just one.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate.from_template(template)

        # Try targeted PDF first when user mentions a specific document
        matched_pdf = self._match_pdf_file(question)
        if matched_pdf:
            context_text = self._get_targeted_pdf_text(matched_pdf, question)
            sources = [matched_pdf]
        else:
            if not self.vector_store:
                self.load_vector_store()
            if not self.vector_store:
                return {"answer": "Vector store not initialized.", "sources": []}
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            source_hint = self._extract_source_hint(question)
            source_docs = retriever.invoke(question)
            source_docs = self._filter_docs_by_source(source_docs, source_hint)
            if not source_docs:
                return {"answer": "Not found in the provided SOPs.", "sources": []}
            context_text = self.format_docs(source_docs)
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in source_docs]))

        if not context_text:
            return {"answer": "Not found in the provided SOPs.", "sources": []}

        # Truncate context to stay within Groq token limits (~4000 tokens for context)
        if len(context_text) > 20000:
            context_text = context_text[:20000]

        rag_chain = (
            {"context": RunnableLambda(lambda _: context_text), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

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

    @staticmethod
    def _sanitize_mermaid(code: str) -> str:
        """Post-process LLM-generated Mermaid code to fix common syntax issues."""
        if not code:
            return ""

        # Remove code fences
        lines = code.strip().split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                continue
            cleaned.append(line)
        code = "\n".join(cleaned).strip()

        # Ensure it starts with flowchart
        if not code.lower().startswith("flowchart"):
            code = "flowchart TD\n" + code

        # Fix node labels: ensure labels with special chars are wrapped in quotes
        # Match patterns like A[label] or A{label} or A(label) and quote the label
        import re as _re

        def _quote_label(m):
            prefix = m.group(1)  # node id
            open_br = m.group(2)  # [ or { or (
            label = m.group(3)    # label text
            close_br = m.group(4) # ] or } or )

            # Remove existing quotes if present, then re-add
            label = label.strip().strip('"').strip("'")
            # Escape any internal double quotes
            label = label.replace('"', "'")
            return f'{prefix}{open_br}"{label}"{close_br}'

        result_lines = []
        for line in code.split("\n"):
            stripped = line.strip()
            # Skip empty lines and the flowchart declaration
            if not stripped or stripped.lower().startswith("flowchart") or stripped.startswith("%%"):
                result_lines.append(line)
                continue

            # Quote labels in node definitions: ID[label], ID{label}, ID(label), ID([label]), ID{{label}}
            # Handle double brackets: ID[["label"]]  and ID{{"label"}}
            line = _re.sub(
                r'(\b\w+)\s*(\[\[|\{\{|\(\[|\[\()(.*?)(\]\]|\}\}|\]\)|\)\])',
                _quote_label, line
            )
            # Handle single brackets: ID[label], ID{label}, ID(label)
            line = _re.sub(
                r'(\b\w+)\s*(\[|\{|\()((?:[^\]\}\)]|\n)*?)(\]|\}|\))',
                _quote_label, line
            )

            result_lines.append(line)

        return "\n".join(result_lines)

    def generate_flowchart(self, question: str) -> Dict[str, Any]:
        """Generates a Mermaid flowchart from SOP context using the LLM."""
        if self.llm is None:
            self.llm = self._setup_llm()

        # Try targeted PDF first for comprehensive content
        matched_pdf = self._match_pdf_file(question)
        if matched_pdf:
            context_text = self._get_targeted_pdf_text(matched_pdf, question)
            sources = [matched_pdf]
        else:
            if not self.vector_store:
                self.load_vector_store()
            if not self.vector_store:
                return {"mermaid": "", "sources": [], "error": "Vector store not initialized."}
            docs = self.retrieve_docs(question, k=5)
            if not docs:
                return {"mermaid": "", "sources": [], "error": "No relevant documents found."}
            context_text = self.format_docs(docs)
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

        if not context_text:
            return {"mermaid": "", "sources": [], "error": "No content found in the document."}

        # Truncate context to stay within Groq token limits
        if len(context_text) > 20000:
            context_text = context_text[:20000]

        template = """You are an expert at creating DETAILED Mermaid flowchart diagrams from SOP documents.

Create a comprehensive Mermaid flowchart from the SOP context below. Include EVERY step, sub-step, and decision point described in the Process Description section.

CRITICAL SYNTAX RULES:
1. Start with: flowchart TD
2. Every node label MUST be wrapped in double quotes. Examples:
   - A["Start Process"]
   - B["Receive Materials"]
   - C{{"Is Quality OK?"}}
   - D["End"]
3. Use square brackets for process steps: A["Step description"]
4. Use double curly braces for decisions: A{{"Decision question?"}}
5. Use round brackets for start/end: A(["Start"]) or A(["End"])
6. Arrows: A --> B for connections, A -->|"Yes"| B for labeled arrows
7. Keep labels SHORT - under 50 chars. No special characters except letters, numbers, spaces, hyphens, and question marks.
8. Do NOT use parentheses, ampersands, slashes, or colons in labels.
9. Do NOT add any text before or after the Mermaid code. No explanations.
10. Do NOT wrap in code fences.

IMPORTANT: Include ALL process steps and sub-steps from the document. Do NOT summarize or skip steps. Each numbered step and sub-step should be a separate node.

Context:
{context}

Question: {question}

flowchart TD"""

        prompt = PromptTemplate.from_template(template)
        chain = (
            {"context": RunnableLambda(lambda _: context_text), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        raw_output = chain.invoke(question).strip()

        # Prepend flowchart TD if the LLM didn't include it
        if not raw_output.lower().startswith("flowchart"):
            raw_output = "flowchart TD\n" + raw_output

        mermaid_code = self._sanitize_mermaid(raw_output)

        return {
            "mermaid": mermaid_code,
            "sources": sources,
            "error": "",
        }

    def generate_table(self, question: str) -> Dict[str, Any]:
        """Generates a markdown table from SOP context using the LLM."""
        if self.llm is None:
            self.llm = self._setup_llm()

        # Try targeted PDF first for comprehensive content
        matched_pdf = self._match_pdf_file(question)
        if matched_pdf:
            context_text = self._get_targeted_pdf_text(matched_pdf, question)
            sources = [matched_pdf]
        else:
            if not self.vector_store:
                self.load_vector_store()
            if not self.vector_store:
                return {"table": "", "sources": [], "error": "Vector store not initialized."}
            docs = self.retrieve_docs(question, k=5)
            if not docs:
                return {"table": "", "sources": [], "error": "No relevant documents found."}
            context_text = self.format_docs(docs)
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

        if not context_text:
            return {"table": "", "sources": [], "error": "No content found in the document."}

        # Truncate context to stay within Groq token limits
        if len(context_text) > 20000:
            context_text = context_text[:20000]

        template = """You are an expert at extracting and organizing data from SOP documents into clear, accurate tables.

Using ONLY the context below, create a well-structured markdown table that answers the question.

Rules:
1. Use ONLY information from the context. Do NOT infer or make up data.
2. Use proper markdown table syntax with headers and alignment.
3. Include ALL rows and columns from the table data found in the context. Do NOT skip any rows.
4. If the context contains a RACI table, reproduce it EXACTLY with all Process steps, Responsible, Accountable, Consulted, and Informed columns.
5. If the context contains multiple related tables (RACI, SIPOC, etc.), present ALL of them.
6. Add a brief title above each table describing what it contains.
7. After the table(s), add a brief note mentioning which SOP document(s) and page(s) the data comes from.
8. Faithfully reproduce the data - do NOT summarize or merge rows.

Context:
{context}

Question: {question}

Markdown Table:"""

        prompt = PromptTemplate.from_template(template)
        chain = (
            {"context": RunnableLambda(lambda _: context_text), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        table_md = chain.invoke(question).strip()
        return {
            "table": table_md,
            "sources": sources,
            "error": "",
        }

    def generate_text(self, prompt: str) -> str:
        """Direct LLM call for auxiliary tasks (e.g., flowchart reconstruction)."""
        if self.llm is None:
            self.llm = self._setup_llm()
        if hasattr(self.llm, "invoke"):
            return self.llm.invoke(prompt)
        return self.llm(prompt)
