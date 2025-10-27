import requests
import json

class OllamaLLM:
    def __init__(self, model="llama3"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):
        payload = {"model": self.model, "prompt": prompt, "stream": True}
        try:
            with requests.post(self.url, json=payload, stream=True) as response:
                if response.status_code != 200:
                    return f"[Error {response.status_code}] {response.text}"

                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            if "response" in data:
                                full_response += data["response"]
                        except json.JSONDecodeError:
                            continue
                return full_response.strip() or "[No output from model]"
        except Exception as e:
            return f"[Connection Error] {e}"

