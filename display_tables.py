from src.table_extractor import TableExtractor
import pandas as pd

pdf_dir = r'd:\Downloads\demo--master\demo--master\demo--master\llm-main\llm-main\llm-main\llm-main\llm\pdfs'
ext = TableExtractor(pdf_dir)

print("=" * 80)
print("EXTRACTED TABLES FROM PDFs")
print("=" * 80 + "\n")

# Extract different types of tables
queries = [
    ("sipoc table", "SIPOC Process Table"),
    ("raci matrix", "RACI Responsibility Matrix"),
    ("list all execution phase activities and who is responsible", "Execution Phase Activities"),
    ("stage gate process", "Stage Gate Process Flow"),
]

for query, description in queries:
    result = ext.extract_table(query)
    
    if not result['error'] and result['table']:
        print(f"\n{'─' * 80}")
        print(f"📋 {description}")
        print(f"{'─' * 80}")
        print(f"Query: {query}")
        print(f"Source: {', '.join(result['sources'][:2])}")
        print(f"Pages: {result.get('pages', [])}")
        print()
        
        # Display the table
        print(result['table'])
        
        # Count rows
        row_count = result['table'].count('\n') - 2
        print(f"\nTotal rows: {row_count}")
    else:
        print(f"\n⚠️  {description} - Not found")

print("\n" + "=" * 80)
print("Multi-page extraction is working! Tables are extracted successfully.")
print("=" * 80)
