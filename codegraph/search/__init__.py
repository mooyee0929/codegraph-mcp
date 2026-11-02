"""Finding symbols: by structure, by text, by vocabulary and (optionally) by meaning.

- :mod:`.tokenize` splits identifiers into words (camelCase, snake_case, paths).
- :mod:`.index` keeps BM25 / TF-IDF postings, persisted by the ``search_index`` pass.
- :mod:`.hybrid` fuses BM25, TF-IDF, FTS5 and embeddings by reciprocal rank.
- :mod:`.embeddings` stores optional sentence-transformers vectors.
- :mod:`.similarity` and :mod:`.related` find symbols that share vocabulary.
- :mod:`.structured` filters and pages nodes by name, kind, place and fan-in/out.
- :mod:`.grep` searches indexed files and names each hit's enclosing symbol.
- :mod:`.snippet` returns one symbol's source.
"""
