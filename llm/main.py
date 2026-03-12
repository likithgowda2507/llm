import argparse
from src.rag_pipeline import SOPRagPipeline

def main():
    parser = argparse.ArgumentParser(description="Quality SOP RAG System")
    parser.add_argument("--index", action="store_true", help="Index the PDF documents")
    parser.add_argument("--query", type=str, help="Ask a question about the SOPs")
    args = parser.parse_args()

    pipeline = SOPRagPipeline(pdf_dir="pdfs")

    if args.index:
        pipeline.load_and_process_documents()
    
    if args.query:
        print(f"\nQuestion: {args.query}")
        result = pipeline.answer_question(args.query)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")

if __name__ == "__main__":
    main()
