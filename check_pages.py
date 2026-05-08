import fitz
pdf = fitz.open('pdfs/SOP - Learning & Development (LDP) UTSFLDPSOP.pdf')
print('Content of pages 7-12 (likely flowchart area):')
for i in range(7, 13):
    text = pdf[i].get_text('text')
    lines = [l.strip() for l in text.split('\n') if l.strip()][:10]
    print(f'\nPage {i}:')
    for line in lines:
        if line:
            print(f'  {line[:70]}')
