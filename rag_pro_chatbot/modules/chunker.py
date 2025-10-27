import nltk
nltk.download('punkt', quiet=True)

def chunk_text(text, chunk_size=512, overlap=20):
    sentences = nltk.sent_tokenize(text)
    chunks, current, count = [], [], 0
    for s in sentences:
        tokens = len(s.split())
        if count + tokens > chunk_size:
            chunks.append(' '.join(current))
            current = current[-overlap:]
            count = sum(len(x.split()) for x in current)
        current.append(s)
        count += tokens
    if current:
        chunks.append(' '.join(current))
    return chunks
