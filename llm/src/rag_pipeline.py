import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class SOPRagPipeline:
    def __init__(self, pdf_dir: str, vector_db_path: str = "faiss_index"):
        self.pdf_dir = pdf_dir
        self.vector_db_path = vector_db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.llm = self._setup_llm()

    def _setup_llm(self):
        """Sets up a local LLM using HuggingFace Pipeline."""
        model_id = "MBZUAI/LaMini-T5-738M"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0,
            do_sample=False
        )
        return HuggingFacePipeline(pipeline=pipe)

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
        return "\n\n".join(doc.page_content for doc in docs)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Retrieves context and generates an answer."""
        if not self.vector_store:
            self.load_vector_store()
        
        if not self.vector_store:
            return {"answer": "Vector store not initialized.", "sources": []}

        template = """Use the following pieces of context to answer the user question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always include the source document name in your final answer.

Context: {context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # LCEL Chain
        rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # For sources, we need to manually retrieve them in LCEL
        source_docs = retriever.invoke(question)
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in source_docs]))
        
        answer = rag_chain.invoke(question)
        
        return {
            "answer": answer,
            "sources": sources
        }
