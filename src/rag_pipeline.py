import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

from src.flowchart_extractor import FlowchartExtractor
from src.table_extractor import TableExtractor
from src.text_pipeline import TextPipeline

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class SOPRagPipeline:
    """Production-ready multi-modal RAG orchestrator.

    Strictly separated pipelines:
    1) Text pipeline    -> Q&A from PDF text chunks
    2) Table pipeline   -> deterministic table extraction (no LLM generation)
    3) Image pipeline   -> flowchart/image extraction from PDF pages
    """

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

        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "")
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "")
        self.groq_base_url = groq_base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

        self.llm = None

        images_dir = str(Path(__file__).resolve().parent.parent / "images")

        self.text_pipeline = TextPipeline(
            pdf_dir=self.pdf_dir,
            vector_db_path=self.vector_db_path,
            llm=None,
        )
        self.table_pipeline = TableExtractor(pdf_dir=self.pdf_dir, llm=None)
        self.flowchart_pipeline = FlowchartExtractor(pdf_dir=self.pdf_dir, images_dir=images_dir, llm=None)

        # Keep compatibility with existing app checks.
        self.vector_store = self.text_pipeline.vector_store

    def _setup_llm(self):
        provider = (self.llm_provider or "").lower().strip()
        if provider == "local":
            return self._setup_local_llm()
        if provider == "groq" or self.groq_api_key:
            return self._setup_groq_llm()
        return self._setup_local_llm()

    def _setup_local_llm(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_id = "MBZUAI/LaMini-T5-738M"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        def generate_text(prompt) -> str:
            if not isinstance(prompt, str):
                prompt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=384, do_sample=False)
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return RunnableLambda(generate_text)

    def _setup_groq_llm(self):
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for Groq API usage.")
        if not self.groq_model:
            raise ValueError("GROQ_MODEL is required for Groq API usage.")

        endpoint = f"{self.groq_base_url.rstrip('/')}/chat/completions"

        def generate_text(prompt) -> str:
            if not isinstance(prompt, str):
                prompt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)

            payload = {
                "model": self.groq_model,
                "messages": [{"role": "user", "content": prompt}],
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

    # ---------- Shared app-facing API ----------
    def load_and_process_documents(
        self,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ):
        status = self.text_pipeline.load_and_process_documents(progress_callback=progress_callback)
        self.vector_store = self.text_pipeline.vector_store
        return status

    def load_vector_store(self):
        self.text_pipeline.load_vector_store()
        self.vector_store = self.text_pipeline.vector_store

    def route_query(self, question: str) -> str:
        q = (question or "").lower()
        if any(k in q for k in ["flowchart", "flow chart", "flow chat", "process flow", "workflow", "diagram"]):
            return "flowchart"

        asks_change_history = any(k in q for k in [
            "change history",
            "revision history",
            "change log",
            "revision log",
            "version history",
        ])
        asks_explicit_table = any(k in q for k in [
            "show table",
            "give table",
            "extract table",
            "display table",
            "tabular",
        ])
        is_fact_question = bool(re.search(r"\b(what|who|when|why|how|which)\b", q))

        # Change-history fact queries should return text answers, not table output,
        # unless the user explicitly asks to display a table.
        if asks_change_history and is_fact_question and not asks_explicit_table:
            return "text"

        if any(k in q for k in [
            "table",
            "tabular",
            "raci",
            "sipoc",
            "matrix",
            "change history",
            "revision history",
            "change log",
            "revision log",
            "version history",
        ]):
            return "table"
        return "text"

    def answer_question(self, question: str) -> Dict[str, Any]:
        if self.llm is None:
            self.llm = self._setup_llm()
            self._propagate_llm()
        return self.text_pipeline.answer_question(question)

    def _propagate_llm(self):
        """Share the initialized LLM with flowchart and table pipelines."""
        self.text_pipeline.llm = self.llm
        self.flowchart_pipeline.llm = self.llm

    def generate_table(self, question: str) -> Dict[str, Any]:
        # Keep table extraction deterministic from PDFs.
        matched_pdf = self._match_pdf_file(question)
        q = (question or "").lower()

        wants_raci = "raci" in q
        wants_sipoc = "sipoc" in q
        wants_change_history = any(k in q for k in ["change history", "revision history", "change log"])
        wants_all_tables = any(k in q for k in ["all tables", "all the tables", "3 tables", "three tables"])

        requested_types = []
        if wants_raci:
            requested_types.append("raci")
        if wants_sipoc:
            requested_types.append("sipoc")
        if wants_change_history:
            requested_types.append("change_history")

        if wants_all_tables and len(requested_types) < 2:
            requested_types = ["raci", "sipoc", "change_history"]

        if len(requested_types) >= 2:
            # Optimize: if both RACI and SIPOC requested, use combined extraction
            if "raci" in requested_types and "sipoc" in requested_types and "change_history" not in requested_types:
                combined_result = self.table_pipeline.extract_raci_and_sipoc(matched_pdf=matched_pdf)
                
                out_tables = []
                
                # Add RACI if found
                if combined_result.get("raci"):
                    raci_data = combined_result["raci"]
                    out_tables.append({
                        "table_type": "raci",
                        "title": "RACI Table",
                        "table": raci_data.get("table", ""),
                        "sources": raci_data.get("sources", []),
                        "error": "",
                    })
                else:
                    out_tables.append({
                        "table_type": "raci",
                        "title": "RACI Table",
                        "table": "",
                        "sources": [],
                        "error": combined_result.get("error", "No RACI table found"),
                    })
                
                # Add SIPOC if found
                if combined_result.get("sipoc"):
                    sipoc_data = combined_result["sipoc"]
                    out_tables.append({
                        "table_type": "sipoc",
                        "title": "SIPOC Table",
                        "table": sipoc_data.get("table", ""),
                        "sources": sipoc_data.get("sources", []),
                        "error": "",
                    })
                else:
                    out_tables.append({
                        "table_type": "sipoc",
                        "title": "SIPOC Table",
                        "table": "",
                        "sources": [],
                        "error": combined_result.get("error", "No SIPOC table found"),
                    })
                
                return {
                    "multi_tables": out_tables,
                    "error": "",
                }
            else:
                # Standard flow for other combinations
                titles = {
                    "raci": "RACI Table",
                    "sipoc": "SIPOC Table",
                    "change_history": "Change History Table",
                }
                out_tables = []
                for table_type in requested_types:
                    t_result = self.table_pipeline.extract_table(
                        question,
                        matched_pdf=matched_pdf,
                        forced_table_type=table_type,
                    )
                    out_tables.append(
                        {
                            "table_type": table_type,
                            "title": titles.get(table_type, table_type.replace("_", " ").title()),
                            "table": t_result.get("table", ""),
                            "sources": t_result.get("sources", []),
                            "error": t_result.get("error", ""),
                        }
                    )

                return {
                    "multi_tables": out_tables,
                    "error": "",
                }

        return self.table_pipeline.extract_table(question, matched_pdf=matched_pdf)

    def generate_flowchart(self, question: str) -> Dict[str, Any]:
        # Ensure LLM is ready before flowchart extraction
        if self.llm is None:
            self.llm = self._setup_llm()
            self._propagate_llm()
        matched_pdf = self._match_pdf_file(question)
        q = (question or "").lower()
        flowchart_mention = bool(re.search(r"flow\s*(chart|charts|chat|flowchart|flowcharts)", q))
        is_overall = bool(
            re.search(r"\bover\s*all\b", q)
            or "overall" in q
        ) and flowchart_mention

        asks_all_flowcharts = bool(
            re.search(r"\b(all|every|complete|entire)\b", q)
            and flowchart_mention
        )
        asks_multiple_flowcharts = bool(
            re.search(r"\b(multiple|many|several|all)\b", q)
            and flowchart_mention
        ) or bool(re.search(r"flow\s*charts\b", q))

        if "particular" in q or "specific" in q or "individual" in q:
            asks_multiple_flowcharts = False

        asks_single_flowchart = bool(
            re.search(r"\b(single|one|only|particular|specific|individual)\b", q)
            and flowchart_mention
        )

        # Overall/all requests should return the full set (up to 6 pages).
        # Specific heading queries should stay focused to one chart (often 1-2 pages).
        if is_overall or asks_all_flowcharts or asks_multiple_flowcharts:
            max_images = 6
        elif asks_single_flowchart:
            max_images = 1
        else:
            max_images = 2
        out = self.flowchart_pipeline.extract_flowcharts(question, matched_pdf=matched_pdf, max_images=max_images)
        return {
            "mermaid": "",
            "sources": out.get("sources", []),
            "error": out.get("error", ""),
            "images": out.get("images", []),
        }

    def _normalize_text(self, text: str) -> str:
        import re

        lowered = (text or "").lower()
        lowered = re.sub(r"\bprocrument\b", "procurement", lowered)
        cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
        return " ".join(cleaned.split())

    def _match_pdf_file(self, question: str):
        q = self._normalize_text(question)
        if not q:
            return None

        root = Path(self.pdf_dir)
        if not root.exists():
            return None

        best = None
        best_score = float("-inf")

        q_tokens = set(q.split())
        stop = {
            "sop", "ut", "for", "of", "and", "the", "in", "a", "an", "to",
            "process", "flow", "chart", "table", "raci", "sipoc", "overall",
            "what", "is", "purpose", "show", "give", "me",
        }

        # Prefer the SOP target phrase explicitly mentioned by the user.
        target_phrase = ""
        m = re.search(r"\bsop\b\s*[-:]?\s*(.+)$", q)
        if m:
            target_phrase = m.group(1).strip()
        if not target_phrase:
            m2 = re.search(r"\bfor\b\s+the\s+(.+)$", q)
            if m2:
                target_phrase = m2.group(1).strip()

        t_tokens = [t for t in target_phrase.split() if t not in stop and len(t) >= 3] if target_phrase else []
        anchor_tokens = t_tokens[:2]
        acronym_tokens = [a.lower() for a in re.findall(r"\(([A-Za-z]{2,8})\)", question or "")]

        for pdf in root.glob("*.pdf"):
            stem = self._normalize_text(pdf.stem)
            if not stem:
                continue

            words = set(stem.split()) - {"sop", "ut", "of", "and", "the", "in", "for", "a", "an", "to"}
            overlap = words & q_tokens
            overlap_score = float(len(overlap))

            phrase_score = 0.0
            if target_phrase:
                if t_tokens:
                    phrase_hits = sum(1 for t in t_tokens if t in stem)
                    phrase_score += phrase_hits * 4.0
                    phrase_norm = " ".join(t_tokens)
                    if phrase_norm and re.search(rf"\b{re.escape(phrase_norm)}\b", stem):
                        phrase_score += 40.0
                    if phrase_hits == len(t_tokens):
                        phrase_score += 8.0

                if anchor_tokens:
                    anchor_hits = sum(1 for t in anchor_tokens if re.search(rf"\b{re.escape(t)}\b", stem))
                    phrase_score += anchor_hits * 6.0
                    missing_anchor = len(anchor_tokens) - anchor_hits
                    if missing_anchor > 0:
                        phrase_score -= missing_anchor * 12.0

            if acronym_tokens:
                for acr in acronym_tokens:
                    if re.search(rf"\b{re.escape(acr)}\b", stem):
                        phrase_score += 18.0
                    else:
                        phrase_score -= 6.0

            contains_full_stem = 2.0 if stem in q else 0.0
            score = overlap_score + phrase_score + contains_full_stem

            # Penalize docs that add many extra topical terms not present in query.
            stem_terms = {w for w in stem.split() if w not in stop and len(w) >= 3}
            q_terms = {w for w in q_tokens if w not in stop and len(w) >= 3}
            extra_terms = stem_terms - q_terms
            score -= min(10.0, float(len(extra_terms)) * 0.75)

            if score > best_score:
                best = pdf.name
                best_score = score

        return best
