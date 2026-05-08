from src.table_extractor import TableExtractor

pdf_dir = r'd:\Downloads\demo--master\demo--master\demo--master\llm-main\llm-main\llm-main\llm-main\llm\pdfs'
ext = TableExtractor(pdf_dir)

# Quick extraction
queries = [
    "sipoc table",
    "raci matrix",
]

for query in queries:
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print('='*70)
    result = ext.extract_table(query)
    if result['error']:
        print(f"Error: {result['error']}")
    else:
        print(f"Source: {result['sources']}")
        print(f"Pages: {result.get('pages', [])}")
        print("\nTable:")
        print(result['table'])
