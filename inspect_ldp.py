import fitz
pdf = fitz.open('pdfs/SOP - Learning & Development (LDP) UTSFLDPSOP.pdf')
print(f'Total pages: {len(pdf)}')
print('\nSearching for process/flow chart pages:')
for i in range(len(pdf)):
    text = pdf[i].get_text('text').lower()
    if any(w in text for w in ['process', 'flow', 'diagram', 'procedure', 'step', 'learner', 'trainer']):
        sample = pdf[i].get_text('text')[:100].replace('\n', ' ')
        print(f'Page {i}: {sample[:70]}')
