from retriever import retrieve
chunks = retrieve('Visa travellers cheques stolen contact number', company='Visa', top_k=5)
for c in chunks:
    print(f"score={c['score']:.4f} | {c['source']} | {c['text'][:120]}")
    print()
