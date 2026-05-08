from src.table_extractor import TableExtractor
import pdfplumber

pdf_dir = r'd:\Downloads\demo--master\demo--master\demo--master\llm-main\llm-main\llm-main\llm-main\llm\pdfs'
pdf_path = r'd:\Downloads\demo--master\demo--master\demo--master\llm-main\llm-main\llm-main\llm-main\llm\pdfs\Stage Gate Process SOP UTSMSGPSOP.pdf'

print("Analyzing SIPOC table structure from raw PDF...\n")

# Check raw structure
with pdfplumber.open(pdf_path) as pdf:
    # Page 11 (0-indexed 10) has 15 rows x 10 cols SIPOC
    page = pdf.pages[10]
    tables = page.extract_tables()
    
    for i, tbl in enumerate(tables):
        if tbl and len(tbl) == 15 and len(tbl[0]) == 10:
            print(f"Found SIPOC table (Table {i}): {len(tbl)} rows x 10 cols\n")
            print("RAW HEADERS:")
            for j, h in enumerate(tbl[0]):
                print(f"  Col {j}: '{h}'")
            
            print("\nRAW DATA (First 3 rows):")
            for row_idx in range(1, min(4, len(tbl))):
                print(f"  Row {row_idx}:", [str(c)[:20] for c in tbl[row_idx]])

print("\n" + "="*80)
print("Extracted via TableExtractor:\n")

ext = TableExtractor(pdf_dir)
result = ext.extract_table("sipoc table")

if not result['error']:
    print("Source:", result['sources'])
    print("Pages:", result.get('pages'))
    print("\nTable:\n")
    print(result['table'])
