# boe-teaching

This repository provides example notebooks demonstrating how to interact with a local [Ollama](https://ollama.ai/) server using Python. Each notebook contains a short introduction and commented code.

## Notebooks

1. **1_simple_starter.ipynb** – basic prompt/answer example using `openai`.
2. **2_structured_output.ipynb** – asks the model for JSON labels on small news items.
3. **3_image_classification.ipynb** – shows how to send a base64 image and get a structured reply.
4. **4_embedding_demo.ipynb** – obtains word embeddings and visualises them.
5. **5_rag_pipeline.ipynb** – minimal retrieval-augmented generation workflow.
6. **6_synthetic_data_classifier.ipynb** – generates synthetic data and trains a classifier.
7. **7_causal_claim_extraction.ipynb** – extracts cause/effect pairs from a large document.

All notebooks rely on common packages such as `openai`, `pandas`, `numpy`, `matplotlib`, `plotly`, and `scikit-learn`.

### Running

Install the optional dependencies and execute a notebook with `jupyter nbconvert`:

```bash
pip install pandas numpy matplotlib plotly scikit-learn pillow openai nbformat nbconvert
jupyter nbconvert --execute notebooks/1_simple_starter.ipynb
```

### Example

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
messages = [{"role": "user", "content": "What is the capital of France?"}]
response = client.chat.completions.create(model="phi4-mini:latest", messages=messages)
print(response.choices[0].message.content.strip())
```

