import pdfplumber

pdf_path = r'd:\Downloads\demo--master\demo--master\demo--master\llm-main\llm-main\llm-main\llm-main\llm\pdfs\Stage Gate Process SOP UTSMSGPSOP.pdf'

print("Raw SIPOC table from PDF:\n")

with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[10]
    tables = page.extract_tables()
    
    for i, tbl in enumerate(tables):
        if tbl and len(tbl) == 15:
            print(f"Table {i}: {len(tbl)} rows x {len(tbl[0])} cols\n")
            print("Headers (10 columns):")
            for j, h in enumerate(tbl[0]):
                print(f"  [{j}]: {str(h)[:40]}")
            print("\nRow 1 data:")
            for j, c in enumerate(tbl[1]):
                print(f"  [{j}]: {str(c)[:40]}")
            break
