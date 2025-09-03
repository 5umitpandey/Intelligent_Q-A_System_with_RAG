**To do:**
1. Update the page number in pre-processing (pages_and_texts.append({"page_number": page_number - 41,  # adjust page numbers since our PDF starts on page 42), so that it works on all the documents.
2. Add a convert_to_pdf so that we can work with emails and other file formats.
3. Use a better model than 7b when better resources arrive.
4. Work on how to avoid missing some tokens when chunk_token_count is MAX.
5. Use Vector database for embedding.

**Topics & Tech to study:**
- Embedding & Transformers
- SBERT.net
- all-mpnet-base-v2 (checkout [inference API](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- NLTK, spaCy
- Nearest Neighbor Search
- Dot product vs Cosine similarity
- Temperature in LLM
- Text sampling

**Notes:**
- Used Regix to solve the : Sentence starting immediately after full stop problem.
