from transformers import pipeline

class Summarizer:
    def __init__(self):
        # Use a robust summarization model (BART or T5)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.max_chunk_words = 800   # avoid exceeding model token limit

    def summarize(self, text, max_len=512):
        # 1️⃣ Handle empty or invalid input
        if not text or len(text.strip()) == 0:
            return "[No text to summarize]"

        # 2️⃣ Split into safe segments if text too long
        words = text.split()
        summaries = []

        for i in range(0, len(words), self.max_chunk_words):
            chunk = " ".join(words[i:i + self.max_chunk_words])

            try:
                # Run summarization on each chunk
                summary = self.summarizer(
                    chunk,
                    max_length=max_len,
                    min_length=60,
                    do_sample=False
                )[0]["summary_text"]
                summaries.append(summary)

            except Exception as e:
                summaries.append(f"[Summarization failed for chunk: {str(e)}]")

        # 3️⃣ Combine partial summaries into final summary
        final_summary = " ".join(summaries)
        return final_summary.strip()

