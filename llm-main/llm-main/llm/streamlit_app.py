import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from PIL import Image
import fitz
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

from src.rag_pipeline import SOPRagPipeline


st.set_page_config(
    page_title="Quality SOP Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', 'Trebuchet MS', 'Segoe UI', sans-serif;
    }
    .stApp {
        background: radial-gradient(1200px 800px at 10% 10%, #1a2445 0%, #0c1020 55%, #0b0f1f 100%);
        color: #e6eefb;
    }
    .block-container {
        padding-top: 2rem;
    }
    .chat-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .chat-subtitle {
        color: #a8b3d3;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

base_dir = Path(__file__).resolve().parent
pdf_dir = str(base_dir / "pdfs")
vector_db = str(base_dir / "faiss_index")
provider = os.getenv("LLM_PROVIDER", "")
model_name = os.getenv("GROQ_MODEL", "")
base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

col_left, col_right = st.columns([1, 1])
with col_left:
    build_index = st.button("Build / Refresh Index")
with col_right:
    refresh_flow = st.button("Rescan Flowcharts")
st.caption("Tip: run index after adding new PDFs.")


def get_pipeline() -> SOPRagPipeline:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = SOPRagPipeline(
            pdf_dir=pdf_dir,
            vector_db_path=vector_db,
            llm_provider=provider,
            groq_model=model_name,
            groq_base_url=base_url,
        )
    return st.session_state.pipeline

if build_index:
    with st.spinner("Indexing PDFs..."):
        get_pipeline().load_and_process_documents()
        st.success("Index built successfully.")

@st.cache_resource(show_spinner=False)
def load_donut():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device

@st.cache_data(show_spinner=False)
def get_flowchart_index(pdf_dir_path: str) -> List[Dict[str, Any]]:
    index = []
    base = Path(pdf_dir_path)
    for pdf in base.glob("*.pdf"):
        try:
            reader = PdfReader(str(pdf))
        except Exception:
            continue
        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if re.search(r"\bflow\s*chart\b", text, re.IGNORECASE):
                index.append({
                    "source": pdf.name,
                    "page": page_index,
                    "text": text,
                })
    return index

if refresh_flow:
    get_flowchart_index.clear()

st.markdown("<div class='chat-title'>Quality SOP Chat Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='chat-subtitle'>Ask questions grounded in your SOP PDFs.</div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about your SOP documents."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def wants_flowchart(question: str) -> bool:
    return bool(re.search(r"\bflow\s*chart\b", question, re.IGNORECASE))

def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    cleaned = []
    for ch in lowered:
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())

def match_pdf_names(question: str, pdf_dir_path: str) -> List[str]:
    question_norm = _normalize_text(question)
    if not question_norm:
        return []
    matches = []
    for pdf in Path(pdf_dir_path).glob("*.pdf"):
        name_norm = _normalize_text(pdf.stem)
        if name_norm and name_norm in question_norm:
            matches.append(pdf.name)
    return matches

def extract_images_from_pdf(pdf_path: Path, max_images: int = 3) -> List[Tuple[Image.Image, str]]:
    results: List[Tuple[Image.Image, str]] = []
    try:
        doc_pdf = fitz.open(str(pdf_path))
    except Exception:
        return results

    for page_index in range(len(doc_pdf)):
        try:
            page_obj = doc_pdf.load_page(page_index)
            images = page_obj.get_images(full=True)
            for img_info in images:
                xref = img_info[0]
                try:
                    img_dict = doc_pdf.extract_image(xref)
                    image_bytes = img_dict.get("image", b"")
                    if not image_bytes:
                        continue
                    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
                    if pil_img.width < 200 or pil_img.height < 200:
                        continue
                    caption = f"{pdf_path.name} (page {page_index + 1})"
                    results.append((pil_img, caption))
                    if len(results) >= max_images:
                        return results
                except Exception:
                    continue
        except Exception:
            continue

    return results

def _extract_json_from_text(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}

def _to_mermaid(nodes: List[str], edges: List[Tuple[str, str]]) -> str:
    if not nodes:
        return ""
    lines = ["flowchart TD"]
    node_ids = {}
    for idx, label in enumerate(nodes, start=1):
        node_id = f"N{idx}"
        node_ids[label] = node_id
        safe_label = label.replace('"', "'")
        lines.append(f"    {node_id}[\"{safe_label}\"]")
    for src, dst in edges:
        if src in node_ids and dst in node_ids:
            lines.append(f"    {node_ids[src]} --> {node_ids[dst]}")
    return "\n".join(lines)

def reconstruct_flowchart(image: Image.Image) -> str:
    try:
        processor, model, device = load_donut()
    except Exception:
        return ""

    image = image.convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    question = (
        "Extract flowchart steps and connections. "
        "Return JSON with keys 'nodes' (list of strings) and 'edges' "
        "(list of [source, target])."
    )
    task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        early_stopping=True,
    )

    seq = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    parsed = _extract_json_from_text(seq)
    nodes = parsed.get("nodes", []) if isinstance(parsed, dict) else []
    edges = parsed.get("edges", []) if isinstance(parsed, dict) else []

    if nodes and edges:
        return _to_mermaid(nodes, edges)

    raw_text = seq.strip()
    if not raw_text:
        return ""

    # Fallback: use the configured LLM to turn raw text into Mermaid.
    prompt = (
        "You are given OCR text from a flowchart image. "
        "Create a Mermaid flowchart TD diagram. "
        "Only output Mermaid code.\n\n"
        f"OCR Text:\n{raw_text}"
    )
    try:
        mermaid = get_pipeline().generate_text(prompt)
        return mermaid.strip()
    except Exception:
        return ""

def render_mermaid(mermaid_code: str) -> None:
    if not mermaid_code:
        return
    html = f"""
    <div class="mermaid">
    {mermaid_code}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
      mermaid.initialize({{ startOnLoad: true }});
    </script>
    """
    st.components.v1.html(html, height=500, scrolling=True)

def extract_flow_images(docs: List, pdf_dir_path: str) -> List[Tuple[Image.Image, str]]:
    results: List[Tuple[Image.Image, str]] = []
    seen = set()
    base = Path(pdf_dir_path)

    for doc in docs:
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", None)
        if not source or page is None:
            continue

        key = (source, page)
        if key in seen:
            continue
        seen.add(key)

        pdf_path = base / source
        if not pdf_path.exists():
            continue

        try:
            doc_pdf = fitz.open(str(pdf_path))
        except Exception:
            continue

        for page_index in [page - 1, page, page + 1]:
            if page_index < 0 or page_index >= len(doc_pdf):
                continue
            try:
                page_obj = doc_pdf.load_page(page_index)
                images = page_obj.get_images(full=True)
                for img_info in images:
                    xref = img_info[0]
                    if (source, page_index, xref) in seen:
                        continue
                    seen.add((source, page_index, xref))
                    try:
                        img_dict = doc_pdf.extract_image(xref)
                        image_bytes = img_dict.get("image", b"")
                        if not image_bytes:
                            continue
                        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
                        if pil_img.width < 200 or pil_img.height < 200:
                            continue
                        caption = f"{source} (page {page_index + 1})"
                        results.append((pil_img, caption))
                        if len(results) >= 3:
                            return results
                    except Exception:
                        continue
            except Exception:
                continue

    return results

prompt = st.chat_input("Ask a question about your SOPs")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if wants_flowchart(prompt):
        with st.chat_message("assistant"):
            with st.spinner("Searching for flowchart images..."):
                matched_pdfs = match_pdf_names(prompt, pdf_dir)
                images: List[Tuple[Image.Image, str]] = []

                if matched_pdfs:
                    for name in matched_pdfs:
                        images.extend(extract_images_from_pdf(Path(pdf_dir) / name, max_images=3))
                        if len(images) >= 3:
                            break
                else:
                    flow_index = get_flowchart_index(pdf_dir)
                    docs = get_pipeline().retrieve_docs(prompt, k=8)
                    relevant_sources = {d.metadata.get("source", "") for d in docs if d.metadata.get("source")}
                    relevant_flows = [f for f in flow_index if f["source"] in relevant_sources]
                    if not relevant_flows:
                        relevant_flows = flow_index

                    for item in relevant_flows[:3]:
                        img_list = extract_flow_images(
                            [type("Doc", (), {"metadata": {"source": item["source"], "page": item["page"]}})],
                            pdf_dir,
                        )
                        images.extend(img_list)
                        if len(images) >= 3:
                            break

                if images:
                    for img, caption in images[:3]:
                        mermaid = reconstruct_flowchart(img)
                        if mermaid:
                            render_mermaid(mermaid)
                        else:
                            st.image(img, caption=caption, use_container_width=True)
                    st.session_state.messages.append({"role": "assistant", "content": "Flowchart images displayed."})
                else:
                    st.markdown("No flowchart images found in the PDFs.")
                    st.session_state.messages.append({"role": "assistant", "content": "No flowchart images found."})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Summarizing relevant PDF text..."):
                result = get_pipeline().summarize_question(prompt, k=6)
                summary = result.get("summary", "Not found in the provided SOPs.")
                sources = result.get("sources", [])
                st.markdown(summary)
                if sources:
                    st.markdown("**Sources:** " + ", ".join(sources))

        st.session_state.messages.append({"role": "assistant", "content": summary})
