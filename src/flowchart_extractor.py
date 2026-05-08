import os
import re
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import importlib


class FlowchartExtractor:
    """
    Image/flowchart extraction pipeline using PyMuPDF.
    - Searches matched PDF first, then falls back to all PDFs.
    - Uses LLM to generate a Mermaid diagram when no image is found.
    - Produces tight, precisely-cropped flowchart images.
    """

    def __init__(self, pdf_dir: str, images_dir: str, llm=None):
        self.pdf_dir = pdf_dir
        self.images_dir = images_dir
        self.llm = llm
        os.makedirs(self.images_dir, exist_ok=True)

    # ─────────────────────────── Public API ───────────────────────────

    def extract_flowcharts(
        self,
        question: str,
        matched_pdf: Optional[str] = None,
        max_images: int = 2,
    ) -> Dict[str, Any]:
        try:
            fitz = importlib.import_module("fitz")
        except Exception:
            return {
                "images": [],
                "mermaid": "",
                "sources": [],
                "error": "PyMuPDF is not installed. Please install pymupdf.",
            }

        # ── Step 1: determine which PDFs to search ──────────────────
        pdf_name = matched_pdf or self._match_pdf_file(question)
        if pdf_name:
            pdf_candidates = [pdf_name]
        else:
            # Search every PDF in directory — broadest fallback
            root = Path(self.pdf_dir)
            pdf_candidates = [p.name for p in sorted(root.glob("*.pdf"))] if root.exists() else []

        if not pdf_candidates:
            return {
                "images": [],
                "mermaid": self._llm_mermaid(question, context=""),
                "sources": [],
                "error": "No PDF documents found.",
            }

        # ── Step 2: extract from each candidate PDF ──────────────────
        keywords = [
            "flowchart", "flow chart", "process flow", "process flow chart",
            "overall process flow", "workflow", "diagram", "process diagram",
            "swimlane", "activity diagram",
        ]
        query_terms = self._query_terms(question)

        if not matched_pdf and len(pdf_candidates) > 1:
            pdf_candidates = self._rank_pdf_candidates(question, pdf_candidates, query_terms)

        images: List[bytes] = []
        sources: List[str] = []
        context_text: str = ""
        q_norm = self._normalize_text(question)
        is_overall = bool(re.search(r"\bover\s*all\b|\boverall\b", q_norm)) and bool(
            re.search(r"flow\s*(chart|chat|flowchart)", q_norm)
        )
        asks_all_flowcharts = bool(
            re.search(r"\b(all|every|multiple|many|several)\b", q_norm)
            and re.search(r"flow\s*(chart|charts|chat|flowchart|flowcharts)", q_norm)
        ) or bool(re.search(r"flow\s*charts\b", q_norm))

        for pdf_name_candidate in pdf_candidates:
            pdf_path = str(Path(self.pdf_dir) / pdf_name_candidate)
            if not os.path.exists(pdf_path):
                continue

            if not is_overall and "management systems" in (pdf_name_candidate or "").lower():
                ms_img, ms_source = self._recover_management_systems_individual_flowchart(
                    fitz,
                    pdf_path,
                    pdf_name_candidate,
                    question,
                )
                if ms_img is not None and ms_source:
                    images.append(ms_img)
                    sources.append(ms_source)
                    break

            result_images, result_sources, result_text = self._extract_from_pdf(
                fitz, pdf_path, pdf_name_candidate, question, keywords, query_terms, max_images - len(images)
            )
            images.extend(result_images)
            sources.extend(result_sources)
            if result_text:
                context_text += result_text + "\n\n"

            if len(images) >= max_images:
                break

        if is_overall and pdf_candidates and "process management" in self._normalize_text(pdf_candidates[0] or ""):
            picked_pages: List[int] = []
            for s in sources:
                m = re.search(r"\(page\s*(\d+)\)", s or "", flags=re.IGNORECASE)
                if m:
                    picked_pages.append(int(m.group(1)))
            bad_toc_pick = bool(picked_pages) and max(picked_pages) <= 3
            if (not images) or bad_toc_pick:
                pmp_imgs, pmp_sources = self._recover_process_management_overall_flowchart(
                    fitz,
                    str(Path(self.pdf_dir) / pdf_candidates[0]),
                    pdf_candidates[0],
                    max_images=max_images,
                )
                if pmp_imgs:
                    images = pmp_imgs
                    sources = pmp_sources

        if (
            (is_overall or asks_all_flowcharts)
            and pdf_candidates
            and "project management" in self._normalize_text(pdf_candidates[0] or "")
            and "industrial" in self._normalize_text(pdf_candidates[0] or "")
            and "solutions" in self._normalize_text(pdf_candidates[0] or "")
            and len(images) < 3
        ):
            pmi_imgs, pmi_sources = self._recover_project_management_industrial_overall_flowcharts(
                fitz,
                str(Path(self.pdf_dir) / pdf_candidates[0]),
                pdf_candidates[0],
                max_images=max_images,
            )
            if pmi_imgs:
                images = pmi_imgs
                sources = pmi_sources

        if (
            (is_overall or asks_all_flowcharts)
            and pdf_candidates
            and "learning" in self._normalize_text(pdf_candidates[0] or "")
            and "development" in self._normalize_text(pdf_candidates[0] or "")
            and len(images) < 4
        ):
            ldp_imgs, ldp_sources = self._recover_learning_development_overall_flowcharts(
                fitz,
                str(Path(self.pdf_dir) / pdf_candidates[0]),
                pdf_candidates[0],
                max_images=max_images,
            )
            if ldp_imgs:
                images = ldp_imgs
                sources = ldp_sources

        # ── Step 3: LLM Mermaid fallback if no images ───────────────
        # SOP-specific rescue: CSV overall flowchart spans two pages and can be
        # under-ranked by text heuristics.
        if (
            is_overall
            and (not images or max((len(b) for b in images), default=0) < 60000)
            and pdf_candidates
            and "computer system validation" in (pdf_candidates[0] or "").lower()
        ):
            csv_imgs, csv_sources = self._recover_csv_overall_flowchart(
                fitz,
                str(Path(self.pdf_dir) / pdf_candidates[0]),
                pdf_candidates[0],
                keywords,
                query_terms,
            )
            if csv_imgs:
                images = csv_imgs[: max_images or 2]
                sources = csv_sources[: max_images or 2]

        if (
            is_overall
            and (not images or max((len(b) for b in images), default=0) < 60000)
            and pdf_candidates
            and "customer escalation" in (pdf_candidates[0] or "").lower()
        ):
            ces_img, ces_source = self._recover_customer_escalation_overall_flowchart(
                fitz,
                str(Path(self.pdf_dir) / pdf_candidates[0]),
                pdf_candidates[0],
            )
            if ces_img is not None and ces_source:
                images = [ces_img]
                sources = [ces_source]

        if (
            is_overall
            and pdf_candidates
            and "management systems" in (pdf_candidates[0] or "").lower()
        ):
            ms_imgs, ms_sources = self._recover_management_systems_overall_flowcharts(
                fitz,
                str(Path(self.pdf_dir) / pdf_candidates[0]),
                pdf_candidates[0],
                max_images=max_images,
            )
            if ms_imgs:
                images = ms_imgs
                sources = ms_sources

        if (
            is_overall
            and (not images)
            and pdf_candidates
            and "procurement" in self._normalize_text(pdf_candidates[0] or "")
        ):
            pro_imgs, pro_sources = self._recover_procurement_overall_flowchart(
                fitz,
                str(Path(self.pdf_dir) / pdf_candidates[0]),
                pdf_candidates[0],
                max_images=max_images,
            )
            if pro_imgs:
                images = pro_imgs
                sources = pro_sources

        if (
            is_overall
            and (not images)
            and pdf_candidates
            and "process management" in self._normalize_text(pdf_candidates[0] or "")
        ):
            pmp_imgs, pmp_sources = self._recover_process_management_overall_flowchart(
                fitz,
                str(Path(self.pdf_dir) / pdf_candidates[0]),
                pdf_candidates[0],
                max_images=max_images,
            )
            if pmp_imgs:
                images = pmp_imgs
                sources = pmp_sources

        mermaid_code = ""
        if not images and self.llm is not None:
            mermaid_code = self._llm_mermaid(question, context_text)

        if not images and not mermaid_code:
            return {
                "images": [],
                "mermaid": "",
                "sources": sources or list({matched_pdf} if matched_pdf else []),
                "error": "No flowchart image found in the document.",
            }

        images, sources = self._sort_images_by_source_page(images, sources)
        return {"images": images, "mermaid": mermaid_code, "sources": sources, "error": ""}

    def _recover_csv_overall_flowchart(
        self,
        fitz: Any,
        pdf_path: str,
        pdf_name: str,
        keywords: List[str],
        query_terms: List[str],
    ) -> Tuple[List[bytes], List[str]]:
        """Recover overall flowchart image for Computer System Validation SOP."""
        if not os.path.exists(pdf_path):
            return [], []

        doc = fitz.open(pdf_path)
        try:
            heading_idx: Optional[int] = None
            for i in range(len(doc)):
                low_raw = (doc[i].get_text("text") or "").lower()
                low = " ".join(low_raw.split())
                # Ignore TOC hits and keep the actual section heading page.
                if "overall process flow chart" in low and "table of contents" not in low:
                    if re.search(r"\b8\s+overall\s+process\s+flow\s+chart\b", low):
                        heading_idx = i
            if heading_idx is None:
                for i in range(len(doc)):
                    low = " ".join((doc[i].get_text("text") or "").lower().split())
                    if "overall process flow chart" in low and i >= 4:
                        heading_idx = i
                        break

            if heading_idx is None:
                return [], []

            candidate_pages: List[int] = []
            if heading_idx < len(doc):
                candidate_pages.append(heading_idx)
            if heading_idx + 1 < len(doc):
                candidate_pages.append(heading_idx + 1)

            images: List[bytes] = []
            sources: List[str] = []
            for idx in candidate_pages:
                page = doc[idx]
                img = self._render_best_crop(fitz, page, keywords, query_terms, focus_terms=None)
                if img is None:
                    page_rect = page.rect
                    clip = fitz.Rect(page_rect.x0 + 8, page_rect.y0 + 8, page_rect.x1 - 8, page_rect.y1 - 8)
                    img = self._pixmap_bytes(fitz, page, clip)
                if img is None:
                    continue
                if len(img) < 12000:
                    continue
                images.append(img)
                sources.append(f"{pdf_name} (page {idx + 1})")

            return images, sources
        finally:
            doc.close()

    def _recover_customer_escalation_overall_flowchart(
        self,
        fitz: Any,
        pdf_path: str,
        pdf_name: str,
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """Recover overall flowchart for Customer Escalation SOP when heading/text and image are split."""
        if not os.path.exists(pdf_path):
            return None, None

        doc = fitz.open(pdf_path)
        try:
            target_idx: Optional[int] = None
            for i in range(len(doc) - 1):
                low = " ".join((doc[i].get_text("text") or "").lower().split())
                if "7 sipoc" in low or "7. sipoc" in low or "overall process flow chart" in low:
                    nxt = i + 1
                    next_diag = self._diagram_signal(fitz, doc[nxt])
                    if next_diag >= 4:
                        target_idx = nxt
                        break

            if target_idx is None:
                return None, None

            page = doc[target_idx]
            page_rect = page.rect
            clip = fitz.Rect(page_rect.x0 + 8, page_rect.y0 + 8, page_rect.x1 - 8, page_rect.y1 - 8)
            img = self._pixmap_bytes(fitz, page, clip)
            if img is None:
                return None, None
            return img, f"{pdf_name} (page {target_idx + 1})"
        finally:
            doc.close()

    def _recover_management_systems_overall_flowcharts(
        self,
        fitz: Any,
        pdf_path: str,
        pdf_name: str,
        max_images: int,
    ) -> Tuple[List[bytes], List[str]]:
        """Recover all six subsection flowcharts for Management Systems SOP (pages 9-14)."""
        if not os.path.exists(pdf_path):
            return [], []

        doc = fitz.open(pdf_path)
        try:
            section_pages = [9, 10, 11, 12, 13, 14]
            images: List[bytes] = []
            sources: List[str] = []

            for p in section_pages:
                if p < 1 or p > len(doc):
                    continue
                page = doc[p - 1]
                img = self._render_best_crop(fitz, page, ["flow chart", "process flow", "workflow"], [], focus_terms=None)
                if img is None:
                    page_rect = page.rect
                    clip = fitz.Rect(page_rect.x0 + 8, page_rect.y0 + 8, page_rect.x1 - 8, page_rect.y1 - 8)
                    img = self._pixmap_bytes(fitz, page, clip)
                if img is None:
                    continue
                if len(img) < 12000:
                    continue
                images.append(img)
                sources.append(f"{pdf_name} (page {p})")
                if len(images) >= max_images:
                    break

            return images, sources
        finally:
            doc.close()

    def _recover_management_systems_individual_flowchart(
        self,
        fitz: Any,
        pdf_path: str,
        pdf_name: str,
        question: str,
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """Recover specific subsection flowchart page for Management Systems SOP."""
        if not os.path.exists(pdf_path):
            return None, None

        q = self._normalize_text(question)
        if not q or ("overall" in q and re.search(r"flow\s*(chart|chat|flowchart)", q)):
            return None, None

        page_map: List[Tuple[List[str], int]] = [
            (["documented information", "creation and updating"], 9),
            (["control of records"], 10),
            (["retained documented"], 10),
            (["internal audit"], 11),
            (["management review", "mrm"], 12),
            (["non conformance", "corrective action", "nonconformance"], 13),
            (["continual improvement"], 14),
        ]

        target_page = None
        for keys, page_no in page_map:
            if any(k in q for k in keys):
                target_page = page_no
                break

        if target_page is None:
            return None, None

        doc = fitz.open(pdf_path)
        try:
            if target_page < 1 or target_page > len(doc):
                return None, None
            page = doc[target_page - 1]
            img = self._render_best_crop(fitz, page, ["flow chart", "process flow", "workflow"], self._query_terms(question), focus_terms=None)
            if img is None:
                page_rect = page.rect
                clip = fitz.Rect(page_rect.x0 + 8, page_rect.y0 + 8, page_rect.x1 - 8, page_rect.y1 - 8)
                img = self._pixmap_bytes(fitz, page, clip)
            if img is None:
                return None, None
            return img, f"{pdf_name} (page {target_page})"
        finally:
            doc.close()

    def _recover_procurement_overall_flowchart(
        self,
        fitz: Any,
        pdf_path: str,
        pdf_name: str,
        max_images: int,
    ) -> Tuple[List[bytes], List[str]]:
        """Recover Procurement overall flowchart when heading and chart are split across pages."""
        if not os.path.exists(pdf_path):
            return [], []

        doc = fitz.open(pdf_path)
        try:
            heading_idx: Optional[int] = None
            for i in range(len(doc)):
                low = " ".join((doc[i].get_text("text") or "").lower().split())
                if "overall process flow chart" in low and "procurement" in low and "table of contents" not in low:
                    heading_idx = i
                    break

            if heading_idx is None:
                return [], []

            candidates = [heading_idx, heading_idx + 1]
            images: List[bytes] = []
            sources: List[str] = []
            for idx in candidates:
                if idx < 0 or idx >= len(doc):
                    continue
                page = doc[idx]
                img = self._render_best_crop(fitz, page, ["flow chart", "process flow", "workflow"], ["procurement"], focus_terms=None)
                if img is None:
                    page_rect = page.rect
                    clip = fitz.Rect(page_rect.x0 + 8, page_rect.y0 + 8, page_rect.x1 - 8, page_rect.y1 - 8)
                    img = self._pixmap_bytes(fitz, page, clip)
                if img is None:
                    continue
                if len(img) < 12000:
                    continue
                images.append(img)
                sources.append(f"{pdf_name} (page {idx + 1})")
                if len(images) >= max_images:
                    break

            return images, sources
        finally:
            doc.close()

    def _recover_process_management_overall_flowchart(
        self,
        fitz: Any,
        pdf_path: str,
        pdf_name: str,
        max_images: int,
    ) -> Tuple[List[bytes], List[str]]:
        """Recover Process Management overall flowchart when heading/chart are split across pages."""
        if not os.path.exists(pdf_path):
            return [], []

        doc = fitz.open(pdf_path)
        try:
            heading_idx: Optional[int] = None
            for i in range(len(doc)):
                raw = doc[i].get_text("text") or ""
                if self._is_skip_page(raw, i):
                    continue
                low = " ".join(raw.lower().split())
                if re.search(r"\b8\s+overall\s+process\s+flow\s+chart\b", low):
                    heading_idx = i
                    break

            for i in range(len(doc)):
                if heading_idx is not None:
                    break
                raw = doc[i].get_text("text") or ""
                if self._is_skip_page(raw, i):
                    continue
                low = " ".join(raw.lower().split())
                if (
                    "overall process flow chart" in low
                    and "process management" in low
                    and "table of contents" not in low
                ):
                    heading_idx = i
                    break

            if heading_idx is None:
                for i in range(len(doc)):
                    raw = doc[i].get_text("text") or ""
                    if self._is_skip_page(raw, i):
                        continue
                    low = " ".join(raw.lower().split())
                    if "overall process flow chart" in low and i >= 4:
                        heading_idx = i
                        break

            if heading_idx is None:
                return [], []

            candidates = [heading_idx, heading_idx + 1]
            images: List[bytes] = []
            sources: List[str] = []
            for idx in candidates:
                if idx < 0 or idx >= len(doc):
                    continue
                page = doc[idx]
                img = self._render_best_crop(
                    fitz,
                    page,
                    ["flow chart", "process flow", "workflow"],
                    ["process", "management"],
                    focus_terms=None,
                )
                if img is None:
                    page_rect = page.rect
                    clip = fitz.Rect(page_rect.x0 + 8, page_rect.y0 + 8, page_rect.x1 - 8, page_rect.y1 - 8)
                    img = self._pixmap_bytes(fitz, page, clip)
                if img is None:
                    continue
                if len(img) < 12000:
                    continue
                images.append(img)
                sources.append(f"{pdf_name} (page {idx + 1})")
                if len(images) >= max_images:
                    break

            return images, sources
        finally:
            doc.close()

    def _recover_project_management_industrial_overall_flowcharts(
        self,
        fitz: Any,
        pdf_path: str,
        pdf_name: str,
        max_images: int,
    ) -> Tuple[List[bytes], List[str]]:
        """Recover the three subsection flowcharts for Project Management Industrial Solutions SOP."""
        if not os.path.exists(pdf_path):
            return [], []

        doc = fitz.open(pdf_path)
        try:
            # Pages contain headings: a) Project Initiation, b) Project Execution, c) Project Closure
            section_pages = [8, 9, 10]
            images: List[bytes] = []
            sources: List[str] = []

            for p in section_pages:
                if p < 1 or p > len(doc):
                    continue
                page = doc[p - 1]
                img = self._render_best_crop(
                    fitz,
                    page,
                    ["flow chart", "process flow", "workflow"],
                    ["project", "management", "industrial", "solutions"],
                    focus_terms=None,
                )
                if img is None:
                    page_rect = page.rect
                    clip = fitz.Rect(page_rect.x0 + 8, page_rect.y0 + 8, page_rect.x1 - 8, page_rect.y1 - 8)
                    img = self._pixmap_bytes(fitz, page, clip)
                if img is None:
                    continue
                if len(img) < 12000:
                    continue
                images.append(img)
                sources.append(f"{pdf_name} (page {p})")
                if len(images) >= max_images:
                    break

            return images, sources
        finally:
            doc.close()

    def _recover_learning_development_overall_flowcharts(
        self,
        fitz: Any,
        pdf_path: str,
        pdf_name: str,
        max_images: int,
    ) -> Tuple[List[bytes], List[str]]:
        """Recover the four flowcharts for Learning & Development SOP (pages 9-12)."""
        if not os.path.exists(pdf_path):
            return [], []

        doc = fitz.open(pdf_path)
        try:
            # Pages 9-12 contain the four main process flowcharts
            section_pages = [9, 10, 11, 12]
            images: List[bytes] = []
            sources: List[str] = []

            for p in section_pages:
                if p < 1 or p > len(doc):
                    continue
                page = doc[p - 1]
                
                # Try render_best_crop first
                img = self._render_best_crop(
                    fitz,
                    page,
                    ["flow chart", "process flow", "workflow", "learning", "development"],
                    ["learning", "development", "training"],
                    focus_terms=None,
                )
                if img is None:
                    # Fallback to full page crop with margins
                    page_rect = page.rect
                    clip = fitz.Rect(page_rect.x0 + 8, page_rect.y0 + 8, page_rect.x1 - 8, page_rect.y1 - 8)
                    img = self._pixmap_bytes(fitz, page, clip)
                
                if img is None:
                    continue
                if len(img) < 12000:
                    continue
                images.append(img)
                sources.append(f"{pdf_name} (page {p})")
                
                # Ensure we capture all 4 pages before breaking
                if p == 12:
                    # Force break after page 12 (we have all 4)
                    break
                if len(images) >= max_images and p < 12:
                    # Only break early if we haven't reached page 12 yet
                    break

            return images, sources
        finally:
            doc.close()

    # ─────────────────────────── PDF Extraction ───────────────────────────

    def _extract_from_pdf(
        self,
        fitz: Any,
        pdf_path: str,
        pdf_name: str,
        question: str,
        keywords: List[str],
        query_terms: List[str],
        max_images: int,
    ) -> Tuple[List[bytes], List[str], str]:
        images: List[bytes] = []
        sources: List[str] = []
        context_text = ""
        selected_pages = set()
        q_norm = self._normalize_text(question)
        prefer_single_page = (
            bool(re.search(r"flow\s*(chart|chat|flowchart)", q_norm))
            and bool(re.search(r"\b(single|one|only|best|main|individual|particular|specific)\b", q_norm))
        )

        doc = fitz.open(pdf_path)
        try:
            ranked_pages = self._rank_flowchart_pages(doc, question, keywords, query_terms)

            requested_heading_terms = self._requested_flow_heading_terms(question)
            if requested_heading_terms and not (
                "overall" in q_norm and bool(re.search(r"flow\s*(chart|chat|flowchart)", q_norm))
            ):
                # Treat explicit heading-target queries as single-chart retrieval.
                prefer_single_page = True
            if requested_heading_terms:
                # Expand candidate pages beyond top-ranked list; some SOPs place heading text
                # and diagrams on neighboring pages that can be under-ranked.
                ranked_map: Dict[int, float] = {p: s for p, s in ranked_pages}
                candidate_pages: List[Tuple[int, float]] = list(ranked_pages)
                for i in range(len(doc)):
                    if i in ranked_map:
                        continue
                    page_i = doc[i]
                    text_i = page_i.get_text("text") or ""
                    if self._is_skip_page(text_i, i):
                        continue
                    diag_i = self._diagram_signal(fitz, page_i)
                    if diag_i < 4:
                        continue
                    if self._looks_like_table_page((text_i or "").lower()) and not self._has_explicit_flow_heading((text_i or "").lower()):
                        continue
                    candidate_pages.append((i, diag_i))

                own_ranked: List[Tuple[int, float]] = []
                prev_ranked: List[Tuple[int, float]] = []
                for page_idx, score in candidate_pages:
                    page_text_norm = self._normalize_text(doc[page_idx].get_text("text") or "")
                    prev_text_norm = ""
                    next_text_norm = ""
                    if page_idx > 0:
                        prev_text_norm = self._normalize_text(doc[page_idx - 1].get_text("text") or "")
                    if page_idx + 1 < len(doc):
                        next_text_norm = self._normalize_text(doc[page_idx + 1].get_text("text") or "")

                    own_match = self._text_has_heading_terms(page_text_norm, requested_heading_terms)
                    prev_match = self._text_has_heading_terms(prev_text_norm, requested_heading_terms)
                    next_match = self._text_has_heading_terms(next_text_norm, requested_heading_terms)

                    curr_diag = self._diagram_signal(fitz, doc[page_idx])
                    next_diag = self._diagram_signal(fitz, doc[page_idx + 1]) if page_idx + 1 < len(doc) else 0.0

                    if own_match:
                        # If the heading page has weak diagram signal and the next page is visual,
                        # prefer the next page (heading -> flowchart split across pages).
                        if page_idx + 1 < len(doc) and next_diag >= 4 and (curr_diag < 6 or next_match):
                            own_ranked.append((page_idx + 1, score + 26.0))
                        if curr_diag >= 4:
                            own_ranked.append((page_idx, score + 22.0))
                    elif prev_match:
                        # Some SOPs place sub-flow heading on previous page and the actual
                        # flowchart drawing on the current page.
                        prev_ranked.append((page_idx, score + 22.0))

                # De-duplicate by keeping best score per page.
                if own_ranked:
                    dedup: Dict[int, float] = {}
                    for p, s in own_ranked:
                        dedup[p] = max(s, dedup.get(p, -1e9))
                    own_ranked = sorted(dedup.items(), key=lambda x: x[1], reverse=True)
                if prev_ranked:
                    dedup: Dict[int, float] = {}
                    for p, s in prev_ranked:
                        dedup[p] = max(s, dedup.get(p, -1e9))
                    prev_ranked = sorted(dedup.items(), key=lambda x: x[1], reverse=True)

                # If any next-page matches exist, prefer them to avoid returning heading pages.
                filtered_ranked = prev_ranked if prev_ranked else own_ranked
                if filtered_ranked:
                    ranked_pages = sorted(filtered_ranked, key=lambda x: x[1], reverse=True)
                else:
                    # Keep default ranking when no specific heading match is found.
                    # This avoids empty results for broad phrases like "individual flow chart".
                    pass

            # ── Primary pass: render cropped page ──
            for page_idx, score in ranked_pages:
                if page_idx in selected_pages:
                    continue
                page = doc[page_idx]
                page_text = page.get_text("text") or ""
                if self._is_skip_page(page_text, page_idx):
                    continue
                page_text_low = page_text.lower()
                explicit_flow_heading = self._has_explicit_flow_heading(page_text_low)
                if self._looks_like_table_page(page_text_low) and not explicit_flow_heading:
                    continue
                likely = self._is_likely_flowchart_page(fitz, page, page_text)
                strong_heading = self._has_strong_flow_heading(page_text)
                # Keep high-confidence heading pages even when visual signal is modest.
                heading_mapped_ok = (
                    bool(requested_heading_terms)
                    and score >= 20
                    and self._diagram_signal(fitz, page) >= 4
                )
                if not likely and not (strong_heading and score >= 18) and not heading_mapped_ok:
                    continue

                target_page_idx = page_idx
                target_page = page
                target_text = page_text

                # Some SOPs place the flowchart heading on one page and the actual
                # rendered chart on the following page with little/no text.
                if (
                    not requested_heading_terms
                    and explicit_flow_heading
                    and (page_idx + 1) < len(doc)
                ):
                    curr_diag = self._diagram_signal(fitz, page)
                    next_idx = page_idx + 1
                    next_page = doc[next_idx]
                    next_text = next_page.get_text("text") or ""
                    next_diag = self._diagram_signal(fitz, next_page)
                    if (
                        curr_diag < 10
                        and next_diag >= 4
                        and not self._is_skip_page(next_text, next_idx)
                        and not self._looks_like_table_page(next_text.lower())
                    ):
                        target_page_idx = next_idx
                        target_page = next_page
                        target_text = next_text

                img_bytes = self._render_best_crop(
                    fitz,
                    target_page,
                    keywords,
                    query_terms,
                    focus_terms=requested_heading_terms,
                )
                if img_bytes is None:
                    continue

                out_path = Path(self.images_dir) / f"{Path(pdf_name).stem}_page_{target_page_idx + 1}.png"
                with open(out_path, "wb") as f:
                    f.write(img_bytes)

                images.append(img_bytes)
                sources.append(f"{pdf_name} (page {target_page_idx + 1})")
                context_text += f"\n[Page {target_page_idx + 1} of {pdf_name}]\n{target_text[:1500]}"
                selected_pages.add(target_page_idx)

                # Some SOP pages contain two flow sections on the same page (e.g., HR/Admin).
                # For overall queries, add a second lower-page crop when multiple section headings appear.
                if len(images) < max_images and not requested_heading_terms:
                    low = page_text.lower()
                    multi_section = len(re.findall(r"compliances?\s*[–-]", low)) >= 2
                    if multi_section:
                        page_rect = page.rect
                        top = max(page_rect.y0 + (page_rect.height * 0.48), self._boilerplate_header_bottom(page))
                        bottom = min(page_rect.y1 - 8, self._boilerplate_footer_top(page))
                        if bottom - top > page_rect.height * 0.18:
                            clip = fitz.Rect(page_rect.x0 + 8, top, page_rect.x1 - 8, bottom)
                            extra_img = self._pixmap_bytes(fitz, page, clip)
                            if extra_img is not None:
                                out_path = Path(self.images_dir) / f"{Path(pdf_name).stem}_page_{page_idx + 1}_part2.png"
                                with open(out_path, "wb") as f:
                                    f.write(extra_img)
                                images.append(extra_img)
                                sources.append(f"{pdf_name} (page {page_idx + 1} - part 2)")

                # If a flowchart continues on the following page, include it.
                if len(images) < max_images and not requested_heading_terms:
                    # Some SOPs place sub-flow headings and chart fragments on the previous page
                    # relative to the strongest-ranked page. Include it when visual signal exists.
                    prev_idx = page_idx - 1
                    if prev_idx >= 0 and prev_idx not in selected_pages:
                        prev_page = doc[prev_idx]
                        prev_text = prev_page.get_text("text") or ""
                        prev_low = prev_text.lower()
                        prev_word_count = len(prev_low.split())
                        prev_diag = self._diagram_signal(fitz, prev_page)
                        prev_has_subheading = bool(re.search(r"\b\d+\.\d+\s+[a-z]", prev_low))
                        if not self._is_skip_page(prev_text, prev_idx):
                            prev_ok = (
                                prev_diag >= 4
                                and not self._looks_like_table_page(prev_low)
                                and not (self._looks_like_text_page(prev_low) and prev_diag < 10)
                                and prev_word_count <= 220
                                and (prev_has_subheading or self._is_likely_flowchart_page(fitz, prev_page, prev_low))
                            )
                            if prev_ok:
                                prev_img_bytes = self._render_best_crop(fitz, prev_page, keywords, query_terms)
                                if prev_img_bytes is not None:
                                    out_path = Path(self.images_dir) / f"{Path(pdf_name).stem}_page_{prev_idx + 1}.png"
                                    with open(out_path, "wb") as f:
                                        f.write(prev_img_bytes)
                                    images.append(prev_img_bytes)
                                    sources.append(f"{pdf_name} (page {prev_idx + 1})")
                                    context_text += f"\n[Page {prev_idx + 1} of {pdf_name}]\n{prev_text[:1500]}"
                                    selected_pages.add(prev_idx)

                    next_idx = page_idx + 1
                    if next_idx < len(doc) and next_idx not in selected_pages:
                        next_page = doc[next_idx]
                        next_text = next_page.get_text("text") or ""
                        if not self._is_skip_page(next_text, next_idx):
                            next_has_heading = self._has_strong_flow_heading(next_text)
                            if self._is_likely_flowchart_page(fitz, next_page, next_text) and not (
                                self._looks_like_table_page(next_text) and not next_has_heading
                            ):
                                next_img_bytes = self._render_best_crop(fitz, next_page, keywords, query_terms)
                                if next_img_bytes is not None:
                                    out_path = Path(self.images_dir) / f"{Path(pdf_name).stem}_page_{next_idx + 1}.png"
                                    with open(out_path, "wb") as f:
                                        f.write(next_img_bytes)
                                    images.append(next_img_bytes)
                                    sources.append(f"{pdf_name} (page {next_idx + 1})")
                                    context_text += f"\n[Page {next_idx + 1} of {pdf_name}]\n{next_text[:1500]}"
                                    selected_pages.add(next_idx)

                if len(images) >= max_images:
                    break

                # Honor explicit single-page asks when users request only one chart page.
                if prefer_single_page and len(images) >= 1:
                    break

            # ── Fallback pass: embedded images ──
            if not images:
                for page_idx, score in ranked_pages:
                    if page_idx in selected_pages:
                        continue
                    page = doc[page_idx]
                    page_text = page.get_text("text") or ""
                    if self._is_skip_page(page_text, page_idx):
                        continue
                    page_text_low = page_text.lower()
                    if self._looks_like_table_page(page_text_low) and not self._has_explicit_flow_heading(page_text_low):
                        continue
                    likely = self._is_likely_flowchart_page(fitz, page, page_text)
                    strong_heading = self._has_strong_flow_heading(page_text)
                    if not likely and not (strong_heading and score >= 18):
                        continue
                    if not (self._has_diagram_signal(fitz, page) or self._has_strong_flow_heading(page_text)):
                        continue

                    for img_no, img in enumerate(page.get_images(full=True)):
                        xref = img[0]
                        base = doc.extract_image(xref)
                        data = base.get("image", b"")
                        if not data:
                            continue
                        w, h = base.get("width", 0), base.get("height", 0)
                        if w < 300 or h < 180:
                            continue

                        out_path = (
                            Path(self.images_dir)
                            / f"{Path(pdf_name).stem}_page_{page_idx + 1}_img_{img_no + 1}.png"
                        )
                        with open(out_path, "wb") as f:
                            f.write(data)

                        images.append(data)
                        sources.append(f"{pdf_name} (page {page_idx + 1})")
                        context_text += f"\n[Page {page_idx + 1} of {pdf_name}]\n{page_text[:1500]}"
                        selected_pages.add(page_idx)

                        if len(images) >= max_images:
                            break
                        # Keep one representative image per page.
                        break
                    if len(images) >= max_images:
                        break
        finally:
            doc.close()

        return images, sources, context_text

    def _sort_images_by_source_page(self, images: List[bytes], sources: List[str]) -> Tuple[List[bytes], List[str]]:
        """Sort image/source pairs by page number so multi-page flowcharts are returned in order."""
        if not images or not sources or len(images) != len(sources):
            return images, sources

        def sort_key(src: str) -> Tuple[int, int, int]:
            page_no = 10**9
            part_no = 1
            m_page = re.search(r"\(page\s*(\d+)", src or "", flags=re.IGNORECASE)
            if m_page:
                page_no = int(m_page.group(1))
            m_part = re.search(r"part\s*(\d+)", src or "", flags=re.IGNORECASE)
            if m_part:
                part_no = int(m_part.group(1))
            return (page_no, part_no, len(src or ""))

        paired = list(zip(images, sources))
        paired.sort(key=lambda item: sort_key(item[1]))
        sorted_images = [p[0] for p in paired]
        sorted_sources = [p[1] for p in paired]
        return sorted_images, sorted_sources

    # ─────────────────────────── Smart Crop ───────────────────────────

    def _render_best_crop(
        self,
        fitz: Any,
        page: Any,
        keywords: List[str],
        query_terms: List[str],
        focus_terms: Optional[List[str]] = None,
    ) -> Optional[bytes]:
        """
        Render the most relevant clipped area of the page at high DPI.
        Uses drawing boxes + image rects + keyword text boxes to compute
        a tight bounding rect. Falls back to full page minus margins.
        """
        page_rect = page.rect
        boilerplate_bottom = self._boilerplate_header_bottom(page)
        boilerplate_top_footer = self._boilerplate_footer_top(page)

        page_text_norm = self._normalize_text(page.get_text("text") or "")
        has_dual_flow_sections = (
            "scrum process flow" in page_text_norm
            and "kanban process flow" in page_text_norm
        )

        # For overall requests on dual-section flow pages, keep the full chart area
        # so both sections are visible.
        if has_dual_flow_sections and not focus_terms:
            clip = fitz.Rect(
                page_rect.x0 + 8,
                max(page_rect.y0 + 8, boilerplate_bottom),
                page_rect.x1 - 8,
                min(page_rect.y1 - 8, boilerplate_top_footer),
            )
            clip &= page_rect
            full_img = self._pixmap_bytes(fitz, page, clip)
            if full_img is not None:
                return full_img

        # Collect drawing bounding boxes (vector shapes)
        draw_boxes = []
        try:
            for d in page.get_drawings():
                r = d.get("rect")
                if r:
                    fr = fitz.Rect(r)
                    if fr.width > 30 and fr.height > 15:
                        draw_boxes.append(fr)
        except Exception:
            pass

        # Collect embedded image bounding boxes
        image_boxes = []
        try:
            imgs = page.get_images(full=True)
            for img in imgs:
                rects = page.get_image_rects(img[0])
                for r in rects:
                    fr = fitz.Rect(r)
                    if fr.width > 100 and fr.height > 60:
                        image_boxes.append(fr)
        except Exception:
            pass

        # If a dominant embedded image exists, it usually is the flowchart body.
        # For specific section requests, prefer heading-anchored crop below.
        if image_boxes and not focus_terms:
            page_area = max(1.0, float(page_rect.width * page_rect.height))
            dominant = max(image_boxes, key=lambda r: float(r.width * r.height))
            dominant_area = float(dominant.width * dominant.height)
            if dominant_area >= page_area * 0.12:
                pad = 10
                clip = fitz.Rect(dominant.x0 - pad, dominant.y0 - pad, dominant.x1 + pad, dominant.y1 + pad)
                clip &= page_rect
                if boilerplate_bottom > clip.y0:
                    clip = fitz.Rect(clip.x0, boilerplate_bottom, clip.x1, clip.y1)
                    clip &= page_rect
                if boilerplate_top_footer < clip.y1:
                    clip = fitz.Rect(clip.x0, clip.y0, clip.x1, boilerplate_top_footer)
                    clip &= page_rect
                if clip.width > page_rect.width * 0.2 and clip.height > page_rect.height * 0.2:
                    img = self._pixmap_bytes(fitz, page, clip)
                    if img is not None:
                        return img

        # Collect keyword-matching text blocks
        text_boxes = []
        focus_heading_boxes: List[Tuple[int, Any]] = []
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            low = (text or "").lower()
            if self._is_boilerplate_text(low):
                continue
            if float(y1) <= (boilerplate_bottom + 4):
                # Ignore header-band matches so query terms in document title do not force full-page crops.
                continue
            if focus_terms:
                norm = self._normalize_text(text or "")
                hits = self._term_match_count(norm, focus_terms)
                if hits >= max(1, len(focus_terms) - 1):
                    focus_heading_boxes.append((hits, fitz.Rect(x0, y0, x1, y1)))
            if any(k in low for k in keywords) or (query_terms and any(t in low for t in query_terms)):
                text_boxes.append(fitz.Rect(x0, y0, x1, y1))

        # For specific flowchart requests, crop only the matched sub-section within the page.
        if focus_heading_boxes:
            focus_heading_boxes.sort(key=lambda it: (-it[0], it[1].y0))
            anchor = focus_heading_boxes[0][1]
            next_top = self._next_flow_subheading_top(page, anchor.y0)
            if next_top is None:
                next_top = self._next_section_heading_top(page, anchor.y0)
            y0 = max(page_rect.y0 + 8, anchor.y0 - 8, boilerplate_bottom)
            y1 = min(page_rect.y1 - 8, boilerplate_top_footer)
            if next_top is not None:
                y1 = min(y1, next_top - 6)
            if y1 - y0 > page_rect.height * 0.12:
                clip = fitz.Rect(page_rect.x0 + 8, y0, page_rect.x1 - 8, y1)
                clip &= page_rect
                focused_img = self._pixmap_bytes(fitz, page, clip)
                if focused_img is not None:
                    return focused_img

        heading_anchor = min(text_boxes, key=lambda r: r.y0) if text_boxes else None
        next_section_top = self._next_section_heading_top(page, heading_anchor.y0 if heading_anchor else None)

        all_boxes = draw_boxes + image_boxes
        if not all_boxes:
            # Only text hints — expand downward from the heading
            if text_boxes:
                heading = heading_anchor
                y1_cap = min(page_rect.y1 - 8, boilerplate_top_footer)
                if next_section_top is not None:
                    y1_cap = min(y1_cap, next_section_top - 8)
                clip = fitz.Rect(
                    page_rect.x0 + 8,
                    max(page_rect.y0, heading.y0 - 10, boilerplate_bottom),
                    page_rect.x1 - 8,
                    y1_cap,
                )
                clip &= page_rect
                if clip.width > page_rect.width * 0.2 and clip.height > page_rect.height * 0.2:
                    return self._pixmap_bytes(fitz, page, clip)
            return None

        # Find heading anchor and prefer boxes below it
        if text_boxes:
            heading = heading_anchor
            below = [r for r in all_boxes if r.y0 >= heading.y0 - 20]
            if below:
                all_boxes = below

        if next_section_top is not None:
            bounded_boxes = []
            for r in all_boxes:
                if r.y0 >= next_section_top - 4:
                    continue
                rr = fitz.Rect(r)
                if rr.y1 > next_section_top - 6:
                    rr = fitz.Rect(rr.x0, rr.y0, rr.x1, next_section_top - 6)
                if rr.width > 20 and rr.height > 10:
                    bounded_boxes.append(rr)
            if bounded_boxes:
                all_boxes = bounded_boxes

        # Union all candidate boxes
        union = all_boxes[0]
        for r in all_boxes[1:]:
            union |= r

        if text_boxes:
            heading = min(text_boxes, key=lambda r: r.y0)
            union |= heading

        # Tight crop with small padding
        pad = 18
        clip = fitz.Rect(
            union.x0 - pad,
            union.y0 - pad,
            union.x1 + pad,
            union.y1 + pad,
        )
        clip &= page_rect
        if boilerplate_bottom > clip.y0:
            clip = fitz.Rect(clip.x0, boilerplate_bottom, clip.x1, clip.y1)
            clip &= page_rect
        if boilerplate_top_footer < clip.y1:
            clip = fitz.Rect(clip.x0, clip.y0, clip.x1, boilerplate_top_footer)
            clip &= page_rect
        if next_section_top is not None and clip.y1 > next_section_top - 6:
            clip = fitz.Rect(clip.x0, clip.y0, clip.x1, next_section_top - 6)
            clip &= page_rect

        # If clip is tiny, use full page
        if clip.width < page_rect.width * 0.12 or clip.height < page_rect.height * 0.12:
            y1_cap = min(page_rect.y1 - 8, boilerplate_top_footer)
            if next_section_top is not None:
                y1_cap = min(y1_cap, next_section_top - 8)
            clip = fitz.Rect(
                page_rect.x0 + 8,
                max(page_rect.y0 + 8, boilerplate_bottom),
                page_rect.x1 - 8,
                y1_cap,
            )

        return self._pixmap_bytes(fitz, page, clip)

    def _next_flow_subheading_top(self, page: Any, start_y: Optional[float]) -> Optional[float]:
        """Find the next in-page flow subheading below an anchor (e.g., Scrum/Kanban headings)."""
        if start_y is None:
            return None

        tops: List[float] = []
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            if y0 <= start_y + 12:
                continue

            line = " ".join((text or "").split())
            if not line:
                continue
            low = line.lower()
            if self._is_boilerplate_text(low):
                continue

            if "process flow" in low or "flow chart" in low or "flowchart" in low:
                tops.append(float(y0))

        if not tops:
            return None
        return min(tops)

    def _next_section_heading_top(self, page: Any, start_y: Optional[float]) -> Optional[float]:
        """Find the top y-position of the next numbered section heading below the flowchart heading."""
        if start_y is None:
            return None

        tops: List[float] = []
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            if y0 <= start_y + 20:
                continue

            line = " ".join((text or "").split())
            if not line:
                continue

            low = line.lower()
            if self._is_boilerplate_text(low):
                continue

            # Example matches: "9 Process Description", "10 References".
            if re.match(r"^\d+\s+[a-z][a-z0-9\s&\-/()]{2,}$", low):
                # Ignore headings that still refer to the flowchart section itself.
                if "flow" in low and "chart" in low:
                    continue
                tops.append(float(y0))
                continue

            # In-page sub-flow headings, e.g. "a) Contracts and Agreements",
            # "1.2 Regulatory Compliance", "b. Employment and Labor Laws".
            if re.match(r"^(?:[a-z][\)\.]|\d+\.\d+)\s+[a-z][a-z0-9\s&\-/()]{2,}$", low):
                tops.append(float(y0))

        if not tops:
            return None
        return min(tops)

    def _pixmap_bytes(self, fitz: Any, page: Any, clip: Any) -> Optional[bytes]:
        """Render a clip at high DPI (3x), convert to PNG bytes."""
        try:
            mat = fitz.Matrix(3.0, 3.0)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            if pix.width < 200 or pix.height < 150:
                # too small — try full page at 2.5x
                pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
            return pix.tobytes("png")
        except Exception:
            return None

    # ─────────────────────────── Page Ranking ───────────────────────────

    def _rank_flowchart_pages(
        self, doc: Any, question: str, keywords: List[str], query_terms: List[str]
    ) -> List[Tuple[int, float]]:
        import importlib
        fitz = importlib.import_module("fitz")

        page_texts = [(doc[i].get_text("text") or "").lower() for i in range(len(doc))]
        page_contexts = [self._neighbor_context(page_texts, i) for i in range(len(doc))]

        term_doc_freq: Dict[str, int] = {}
        for term in query_terms:
            term_doc_freq[term] = sum(1 for ctx in page_contexts if term in ctx)

        focus_phrase = self._focus_phrase(question)
        
        ranked: List[Tuple[int, float]] = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page_texts[page_idx]
            context_text = page_contexts[page_idx]
            word_count = len(text.split())
            
            if self._is_skip_page(text, page_idx):
                continue

            keyword_hits = sum(1 for k in keywords if k in text)
            
            # Query score uses neighboring context to catch titles at page boundaries.
            query_present = [t for t in query_terms if t in context_text]
            query_hits = len(query_present)
            query_score = 0.0
            for t in query_present:
                df = term_doc_freq.get(t, 1)
                query_score += 1.0 + (len(doc) / max(df, 1)) * 0.35

            if query_terms:
                coverage = query_hits / max(len(query_terms), 1)
                if coverage >= 0.45:
                    query_score += 18
                elif coverage >= 0.25:
                    query_score += 8
                elif coverage == 0:
                    query_score -= 12

            if focus_phrase and focus_phrase in context_text:
                query_score += 14

            diagram_signal = self._diagram_signal(fitz, page)
            strong_heading = self._has_strong_flow_heading(text)
            explicit_heading = self._has_explicit_flow_heading(text)

            # Weight visual features and explicit keywords heavily
            score = (keyword_hits * 6) + query_score + diagram_signal

            # Heavily penalize RACI tables / Matrices. A RACI matrix table grid
            # is drawn with lines, which artificially balloons the diagram_signal.
            if re.search(r"\braci\b", text):
                score -= 50
            elif "responsible" in text and "accountable" in text and "consulted" in text:
                score -= 50
            elif sum(1 for w in ["matrix", "table", "role", "responsibility"] if w in text) >= 3:
                score -= 20

            # Penalize pages lacking visual content
            if diagram_signal < 4:
                score -= 10

            # Strongly down-rank prose-heavy pages to avoid text-image outputs.
            if self._looks_like_text_page(text) and diagram_signal < 22:
                score -= 30

            # Down-rank table pages so RACI/SIPOC grids are not returned for flowchart queries.
            if self._looks_like_table_page(text) and not strong_heading:
                score -= 60

            # Image-heavy pages with only boilerplate text often correspond to annexures.
            # Require explicit flow heading for such sparse pages.
            if word_count <= 95 and not explicit_heading:
                score -= 40

            if not self._is_likely_flowchart_page(fitz, page, text):
                score -= 45
                
            if strong_heading:
                score += 15

            if score <= 0:
                continue

            ranked.append((page_idx, score))

        # ── Title-page lookahead: "Title on page N, flowchart on page N+1" fix ──
        # For each page with high query relevance but low diagram signal (a title heading page),
        # propagate 75% of its score to the next page that actually has the diagram.
        raw_scores: Dict[int, float] = {p: s for p, s in ranked}
        for page_idx, score in list(raw_scores.items()):
            page = doc[page_idx]
            ds = self._diagram_signal(fitz, page)
            # Identify title pages: good relevance but weak visual diagram signal
            if ds < 12 and score > 15:
                for offset in [1, 2]:
                    next_idx = page_idx + offset
                    if next_idx >= len(doc):
                        break
                    next_page = doc[next_idx]
                    next_text = page_texts[next_idx]
                    next_ds = self._diagram_signal(fitz, next_page)
                    if self._is_skip_page(next_text, next_idx):
                        break
                    # If next page has a real diagram, boost it with the title page's relevance
                    if next_ds >= 10:
                        carry = score * 0.75
                        raw_scores[next_idx] = raw_scores.get(next_idx, 0.0) + carry
                        break  # only propagate to the first visual page

        ranked = [(p, s) for p, s in raw_scores.items() if s > 0]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:10]

    def _neighbor_context(self, page_texts: List[str], page_idx: int) -> str:
        prev_tail = page_texts[page_idx - 1][-1200:] if page_idx > 0 else ""
        cur = page_texts[page_idx]
        next_head = page_texts[page_idx + 1][:1200] if page_idx + 1 < len(page_texts) else ""
        return f"{prev_tail}\n{cur}\n{next_head}"

    def _focus_phrase(self, question: str) -> str:
        q = self._normalize_text(question)
        if not q:
            return ""
        m = re.search(r"\b(?:for|of|about|on)\b\s+(.+)$", q)
        phrase = m.group(1).strip() if m else q
        tokens = [
            t
            for t in phrase.split()
            if t not in {
                "show", "give", "me", "the", "a", "an", "flow", "chart", "flowchart",
                "process", "diagram", "workflow", "sop", "please", "overall",
            }
        ]
        if len(tokens) < 2:
            return ""
        return " ".join(tokens[:7])

    def _looks_like_text_page(self, text: str) -> bool:
        low = (text or "").lower()
        words = low.split()
        if len(words) < 120:
            return False

        flow_terms = [
            "flowchart", "flow chart", "process flow", "workflow", "decision", "start", "end", "step",
        ]
        flow_hits = sum(1 for t in flow_terms if t in low)
        paragraph_markers = sum(low.count(m) for m in ["\n-", "\n*", "\n•", " shall ", " should ", " procedure "])
        dense_lines = sum(1 for ln in low.splitlines() if len(ln.split()) >= 10)

        # Large prose content with weak flow markers is likely a text section page.
        return flow_hits <= 1 and (paragraph_markers >= 2 or dense_lines >= 6)

    def _looks_like_table_page(self, text: str) -> bool:
        low = (text or "").lower()
        if not low.strip():
            return False

        # Strong schema signatures should always be treated as table pages.
        sipoc_signature = (
            "supplier" in low
            and "input" in low
            and "process" in low
            and "output" in low
            and "customer" in low
        )
        if sipoc_signature:
            return True

        raci_signature = (
            "responsible" in low
            and "accountable" in low
            and "consulted" in low
            and "informed" in low
        )
        if raci_signature:
            return True

        if "sipoc" in low and sum(1 for t in ["supplier", "input", "process", "output", "customer"] if t in low) >= 3:
            return True
        if "raci" in low and sum(1 for t in ["responsible", "accountable", "consulted", "informed"] if t in low) >= 2:
            return True

        flow_hits = sum(
            1 for t in ["flowchart", "flow chart", "process flow", "workflow", "overall process flow"]
            if t in low
        )
        table_hits = sum(
            1
            for t in [
                "raci", "sipoc", "responsible", "accountable", "consulted", "informed",
                "supplier", "input", "output", "customer", "change history",
                "document version", "reason for change", "sl. no", "matrix", "table",
            ]
            if t in low
        )

        if table_hits >= 5 and flow_hits == 0:
            return True
        if table_hits >= 7 and flow_hits <= 1:
            return True
        return False

    def _is_likely_flowchart_page(self, fitz: Any, page: Any, text: str) -> bool:
        low = (text or "").lower()
        words = low.split()
        word_count = len(words)
        dense_lines = sum(1 for ln in low.splitlines() if len(ln.split()) >= 10)
        flow_term_hits = sum(1 for t in ["flowchart", "flow chart", "process flow", "workflow", "start", "end", "decision"] if t in low)
        diagram_signal = self._diagram_signal(fitz, page)
        strong_heading = self._has_strong_flow_heading(low)

        if self._looks_like_table_page(low) and not strong_heading:
            return False

        # Strong visual diagrams should pass even if title text is sparse.
        if diagram_signal >= 24:
            return True

        # Typical flowchart pages: visual signal + not prose-heavy.
        if diagram_signal >= 12 and dense_lines <= 6 and word_count <= 220:
            return True

        # Explicit flow heading with at least some diagram signal.
        if strong_heading and diagram_signal >= 6:
            return True

        # Some SOPs render flowcharts with low vector counts.
        if strong_heading and diagram_signal >= 4 and word_count <= 350:
            return True

        if flow_term_hits >= 1 and diagram_signal >= 5 and word_count <= 260:
            return True

        # Reject prose-heavy pages that may include incidental boxes/lines.
        if self._looks_like_text_page(low) and diagram_signal < 24:
            return False
        if word_count > 240 and dense_lines >= 8 and flow_term_hits <= 1:
            return False

        return flow_term_hits >= 2 and diagram_signal >= 8

    def _rank_pdf_candidates(self, question: str, pdf_names: List[str], query_terms: List[str]) -> List[str]:
        """Sort PDFs so the most question-relevant documents are searched first."""
        try:
            fitz = importlib.import_module("fitz")
        except Exception:
            return pdf_names

        focus_phrase = self._focus_phrase(question)
        q_tokens = set(self._normalize_text(question).split())
        filename_stop = {"sop", "ut", "v", "issue", "process", "flow", "chart", "training"}

        scored: List[Tuple[str, float]] = []
        for pdf_name in pdf_names:
            pdf_path = Path(self.pdf_dir) / pdf_name
            if not pdf_path.exists():
                continue

            score = 0.0

            # File-name overlap provides a cheap initial hint.
            stem_tokens = set(self._normalize_text(Path(pdf_name).stem).split()) - filename_stop
            score += len(stem_tokens & q_tokens) * 4.0

            doc = None
            try:
                doc = fitz.open(str(pdf_path))
                text_chunks: List[str] = []
                for i in range(len(doc)):
                    text_chunks.append((doc[i].get_text("text") or "").lower())
                preview = "\n".join(text_chunks)

                if focus_phrase and focus_phrase in preview:
                    score += 24.0

                for term in query_terms:
                    if term in preview:
                        score += 3.0

                if "flow" in preview and "chart" in preview:
                    score += 4.0
            except Exception:
                pass
            finally:
                if doc is not None:
                    doc.close()

            scored.append((pdf_name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scored] if scored else pdf_names

    # ─────────────────────────── LLM Mermaid Fallback ───────────────────────────

    def _llm_mermaid(self, question: str, context: str) -> str:
        """Ask the LLM to generate a Mermaid flowchart from text context."""
        if self.llm is None:
            return ""
        try:
            ctx = context.strip()[:4000] if context else ""
            prompt = (
                "You are an expert at creating Mermaid.js flowchart diagrams from SOP documents.\n\n"
                "Based on the user question and SOP context below, generate a valid Mermaid flowchart diagram.\n"
                "Rules:\n"
                "- Use `graph TD` (top-down) layout.\n"
                "- Include clear, labeled nodes and arrows.\n"
                "- Only describe steps/processes found in the context.\n"
                "- If context is empty, generate a reasonable process diagram based on the question.\n"
                "- Output ONLY the raw Mermaid code, no markdown fences, no explanation.\n\n"
                f"User Question: {question}\n\n"
                f"SOP Context:\n{ctx if ctx else 'No context available.'}\n\n"
                "Mermaid Code:"
            )
            result = self.llm.invoke(prompt)
            code = str(result).strip()
            # Strip markdown fences if present
            code = re.sub(r"^```(?:mermaid)?\s*", "", code, flags=re.IGNORECASE)
            code = re.sub(r"\s*```$", "", code)
            code = code.strip()
            if "graph" not in code.lower() and "flowchart" not in code.lower():
                return ""
            return code
        except Exception:
            return ""

    # ─────────────────────────── Helpers ───────────────────────────

    def _is_skip_page(self, text: str, page_idx: int) -> bool:
        low = (text or "").lower()
        if not low.strip():
            return True

        # Title / Cover pages often have very few words or distinct markers
        word_count = len(low.split())
        if page_idx == 0 and (word_count < 80 or "document title" in low or "document no" in low):
            return True

        # Table of contents
        if "table of contents" in low or re.search(r"\btable of content\b", low):
            return True
        if re.search(r"^\s*contents\b", low, flags=re.MULTILINE):
            return True

        # Revision / Change history (these are typically text/table data, not visual flowcharts)
        if "revision history" in low or "change history" in low or "document history" in low:
            # ONLY skip if it doesn't also prominently mention flowchart words
            if "flowchart" not in low and "diagram" not in low:
                return True

        dotted = len(re.findall(r"\.{3,}", low))
        lines = [ln.strip() for ln in low.splitlines() if ln.strip()]
        line_num_endings = sum(1 for ln in lines if re.search(r"\b\d{1,3}\s*$", ln))
        if dotted >= 4 and line_num_endings >= 4:
            return True
        return False

    def _diagram_signal(self, fitz: Any, page: Any) -> float:
        draw_count = 0
        try:
            for d in page.get_drawings():
                r = d.get("rect")
                if r:
                    fr = fitz.Rect(r)
                    if fr.width > 20 and fr.height > 20:
                        draw_count += 1
        except Exception:
            pass

        img_score = 0
        try:
            for img in page.get_images(full=True):
                rects = page.get_image_rects(img[0])
                for r in rects:
                    fr = fitz.Rect(r)
                    if fr.width > 250 and fr.height > 150:
                        img_score += 15
                    elif fr.width > 100 and fr.height > 60:
                        img_score += 5
        except Exception:
            pass

        return min(draw_count, 60) * 0.5 + img_score

    def _has_diagram_signal(self, fitz: Any, page: Any) -> bool:
        return self._diagram_signal(fitz, page) >= 6

    def _has_strong_flow_heading(self, text: str) -> bool:
        low = (text or "").lower()
        patterns = [
            "overall process flow", "process flow chart", "flow chart",
            "workflow", "process diagram", "activity diagram", "swimlane",
        ]
        if any(p in low for p in patterns):
            return True

        # Many SOPs split the overall flow into labeled sub-sections like
        # "a) GRN Process", "b) Inventory Handling", "c) Stock Transfer Process".
        if re.search(r"\b[a-z]\)\s*[a-z0-9/&\-\s]{2,40}\b(process|handling|transfer|grn)\b", low):
            return True

        return False

    def _has_explicit_flow_heading(self, text: str) -> bool:
        low = (text or "").lower()
        if "overall process flow" in low:
            return True
        if "process flow chart" in low:
            return True
        if re.search(r"\bflow\s*chart\b", low):
            return True
        return False

    def _is_boilerplate_text(self, text: str) -> bool:
        low = (text or "").lower()
        keys = [
            "document title", "document no", "effective date", "next review",
            "version", "issue", "document classification", "document status",
            "document template", "confidential", "classified", "page ",
        ]
        return any(k in low for k in keys)

    def _boilerplate_header_bottom(self, page: Any) -> float:
        """Return y-coordinate below detected SOP header boilerplate."""
        page_rect = page.rect
        cutoff = page_rect.y0
        max_scan_y = page_rect.y0 + (page_rect.height * 0.35)

        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            if y0 > max_scan_y:
                continue
            low = (text or "").lower()
            if self._is_boilerplate_text(low):
                cutoff = max(cutoff, float(y1) + 10)

        return min(cutoff, page_rect.y0 + page_rect.height * 0.5)

    def _boilerplate_footer_top(self, page: Any) -> float:
        """Return y-coordinate above detected footer boilerplate."""
        page_rect = page.rect
        cutoff = page_rect.y1
        min_scan_y = page_rect.y0 + (page_rect.height * 0.65)

        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            if y1 < min_scan_y:
                continue
            low = (text or "").lower()
            if self._is_boilerplate_text(low):
                cutoff = min(cutoff, float(y0) - 8)
                continue
            if re.search(r"\bpage\s*\d+(?:\s*of\s*\d+)?\b", low):
                cutoff = min(cutoff, float(y0) - 8)
                continue
            if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", low):
                cutoff = min(cutoff, float(y0) - 8)

        return max(cutoff, page_rect.y0 + page_rect.height * 0.45)

    def _query_terms(self, question: str) -> List[str]:
        q = self._normalize_text(question)
        stop = {
            "show", "give", "me", "the", "a", "an", "for", "of", "to", "from", "in",
            "on", "flow", "chart", "flowchart", "process", "diagram", "workflow",
            "sop", "please", "what", "how", "does", "is", "are", "get",
        }
        return [w for w in q.split() if len(w) >= 3 and w not in stop]

    def _requested_flow_heading_terms(self, question: str) -> List[str]:
        q = self._normalize_text(question)
        if not q:
            return []

        # Overall flowchart requests should keep broad page selection/crop.
        if "overall" in q and re.search(r"flow\s*(chart|chat|flowchart)", q):
            return []

        # Strong individual selectors should map to a single section term.
        if re.search(r"\bindividual\b|\bparticular\b|\bspecific\b", q):
            individual_slice = ""
            m_ind = re.search(r"(?:individual|particular|specific)\s+(.+?)\s+flow\s*(?:chart|chat|flowchart)", q)
            if m_ind:
                individual_slice = m_ind.group(1)
            target = individual_slice or q

            if "kanban" in target:
                return ["kanban"]
            if "scrum" in target:
                return ["scrum"]

        # Prefer explicit qualifier text after separators like "- project execution".
        raw_q = (question or "").strip().lower()
        phrase = ""
        dash_parts = [p.strip() for p in re.split(r"\s[-:|]\s", raw_q) if p.strip()]
        if len(dash_parts) >= 2:
            phrase = dash_parts[-1]

        # Prefer specific text before "flow chart" for queries such as:
        # "Vendor Management flow chart in SOP - Learning & Development".
        before_phrase = ""
        m_before = re.search(r"(.+?)\bflow\s*(?:chart|chat|flowchart)\b", q)
        if m_before:
            before_phrase = m_before.group(1).strip()
            before_phrase = re.sub(r"^(show|give|get|display|extract)\s+", "", before_phrase).strip()
            before_phrase = re.sub(r"\b(for|of|on|in)\s+the\s+sop\b.*$", "", before_phrase).strip()
            before_phrase = re.sub(r"\bfor\s+sop\b.*$", "", before_phrase).strip()
            if re.search(r"\b(sop|flow|chart|process|workflow)\b", before_phrase) and len(before_phrase.split()) <= 1:
                before_phrase = ""
        if before_phrase:
            phrase = before_phrase

        # Otherwise, capture text after "flow chart" when present.
        if not phrase:
            m_after = re.search(r"\bflow\s*(?:chart|chat|flowchart)\b\s*(?:for|of|on|in)?\s*(.+)$", q)
            if m_after:
                phrase = m_after.group(1).strip()

        # Fallback: use whole question only if it contains strong section qualifiers.
        if not phrase:
            if any(k in q for k in ["initiation", "execution", "closure", "phase", "step"]):
                phrase = q
            else:
                return []

        generic = {
            "overall", "process", "flow", "chart", "flowchart", "workflow", "diagram",
            "show", "give", "me", "the", "a", "an", "for", "of", "to", "in", "on", "sop",
            "over", "all", "chat",
            "individual", "particular", "specific", "separate", "single", "one",
            "statutory", "regulatory", "compliance", "compliances", "monitoring", "reporting",
            "project", "industrial", "solutions",
            "learning", "development",
        }
        spell_map = {
            "defination": "definition",
            "definitaion": "definition",
            "definision": "definition",
            "improvment": "improvement",
        }
        terms = []
        for t in phrase.split():
            if len(t) < 3 or t in generic:
                continue
            terms.append(spell_map.get(t, t))

        # Keep only meaningful section qualifiers when available.
        priority_terms = [
            t for t in terms
            if t in {"initiation", "execution", "closure", "planning", "approval", "design", "implementation", "testing", "deployment"}
        ]
        if priority_terms:
            terms = priority_terms

        # Avoid overly strict matching for long phrases while preserving specific qualifiers
        # that often appear at the end (e.g., finance/hr/admin).
        if len(terms) > 4:
            terms = terms[:3] + terms[-1:]
        return terms

    def _text_has_heading_terms(self, text_norm: str, terms: List[str]) -> bool:
        if not terms:
            return False
        if not text_norm:
            return False

        tokens = set(text_norm.split())

        def fuzzy_present(term: str) -> bool:
            if term in tokens:
                return True
            if term in text_norm:
                return True
            # lightweight typo tolerance for heading terms
            for tok in tokens:
                if abs(len(tok) - len(term)) > 2:
                    continue
                mismatches = 0
                for a, b in zip(tok, term):
                    if a != b:
                        mismatches += 1
                        if mismatches > 2:
                            break
                mismatches += abs(len(tok) - len(term))
                if mismatches <= 2:
                    return True
            return False

        # Require all terms for short specific requests, but allow partial match
        # for longer phrases to improve OCR robustness.
        hits = sum(1 for t in terms if fuzzy_present(t))
        if len(terms) <= 2:
            return hits == len(terms)
        return hits >= max(2, len(terms) - 1)

    def _term_match_count(self, text_norm: str, terms: List[str]) -> int:
        if not terms or not text_norm:
            return 0
        return sum(1 for t in terms if t in text_norm)

    def _normalize_text(self, text: str) -> str:
        lowered = (text or "").lower()
        lowered = re.sub(r"\bprocrument\b", "procurement", lowered)
        return " ".join(re.sub(r"[^a-zA-Z0-9\s]", " ", lowered).split())

    def _match_pdf_file(self, question: str) -> Optional[str]:
        q = self._normalize_text(question)
        if not q:
            return None
        root = Path(self.pdf_dir)
        if not root.exists():
            return None

        best = None
        best_score = float("-inf")

        q_tokens = set(q.split())
        stop = {"sop", "ut", "for", "of", "and", "the", "in", "a", "an", "to", "process", "flow", "chart", "table"}

        target_phrase = ""
        m = re.search(r"\bsop\b\s*[-:]?\s*(.+)$", q)
        if m:
            target_phrase = m.group(1).strip()
        if not target_phrase:
            m2 = re.search(r"\bfor\b\s+the\s+(.+)$", q)
            if m2:
                target_phrase = m2.group(1).strip()

        for pdf in root.glob("*.pdf"):
            stem = self._normalize_text(pdf.stem)
            if not stem:
                continue

            words = set(stem.split()) - {"sop", "ut", "of", "and", "the", "in", "for", "a", "an", "to"}
            overlap = words & q_tokens
            overlap_score = float(len(overlap))

            phrase_score = 0.0
            if target_phrase:
                t_tokens = [t for t in target_phrase.split() if t not in stop and len(t) >= 3]
                if t_tokens:
                    phrase_hits = sum(1 for t in t_tokens if t in stem)
                    phrase_score += phrase_hits * 4.0
                    if phrase_hits == len(t_tokens):
                        phrase_score += 8.0

            contains_full_stem = 2.0 if stem in q else 0.0
            score = overlap_score + phrase_score + contains_full_stem

            if score > best_score:
                best = pdf.name
                best_score = score

        return best
