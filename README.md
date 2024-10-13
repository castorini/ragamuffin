# RAGamuffin: The Redis of IR

If you have $O(100K)$ documents, own a beefy GPU, and want a fluent and efficient information retrieval toolkit that "just works," then RAGamuffin is for you!

## Code Sample

```python
import ragamuffin as rg

documents = [...]
encode_pipeline = rg.Chunkify() | rg.Tokenize() | (rg.DenseDocumentEncoder() & rg.BM25())
db = encode_pipeline(documents)

query_pipeline = rg.Chunkify() | rg.Tokenize() | (rg.DenseQueryEncoder() & rg.BM25()) | db | TopK(10)
relevant_documents = query_pipeline("...")
```
