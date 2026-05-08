import fitz
pdf = fitz.open('pdfs/SOP - Learning & Development (LDP) UTSFLDPSOP.pdf')
print('Pages with diagram/low-text content (likely flowchart pages):')
diagram_pages = []
for i in range(len(pdf)):
    page = pdf[i]
    text = page.get_text('text')
    images = page.get_images()
    # Diagram pages usually have few text lines but rich graphics
    if len(text) < 800 or len(images) > 0:
        diagram_pages.append(i)
        print(f'Page {i}: text_len={len(text)}, images={len(images)}')
        
print(f'\nDiagram pages found: {diagram_pages}')
