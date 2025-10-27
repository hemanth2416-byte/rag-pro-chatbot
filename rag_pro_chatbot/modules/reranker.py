from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs):
        pairs = [[query, d] for d in docs]
        scores = self.model.predict(pairs)
        ranked_docs = [d for _, d in sorted(zip(scores, docs), reverse=True)]
        return ranked_docs

