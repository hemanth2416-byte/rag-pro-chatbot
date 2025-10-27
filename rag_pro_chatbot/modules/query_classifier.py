from transformers import pipeline

class QueryClassifier:
    def __init__(self):
        # Use zero-shot classification pipeline for MNLI model
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def needs_retrieval(self, query: str) -> bool:
        """
        Classify if the query needs retrieval.
        """
        candidate_labels = ["needs retrieval", "no retrieval needed"]
        result = self.classifier(query, candidate_labels)
        label = result["labels"][0]
        return label == "needs retrieval"

