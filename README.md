# boe-teaching

This repository provides example notebooks demonstrating how to interact with a local [Ollama](https://ollama.ai/) server using Python. Each notebook contains a short introduction, commented code and simulated data so they can run without external resources.

## Notebooks

1. **1_simple_starter.ipynb** – basic prompt/answer example using `requests`.
2. **2_structured_output.ipynb** – asks the model for JSON labels on small news items.
3. **3_image_classification.ipynb** – shows how to send a base64 image and get a structured reply.
4. **4_embedding_demo.ipynb** – obtains word embeddings and visualises them.
5. **5_rag_pipeline.ipynb** – minimal retrieval-augmented generation workflow.
6. **6_synthetic_data_classifier.ipynb** – generates synthetic data and trains a classifier.
7. **7_causal_claim_extraction.ipynb** – extracts cause/effect pairs from a large document.
8. **8_planning_classifier_pipeline.ipynb** – pipeline for embedding and classifying planning descriptions.
9. **minimal_rag.ipynb** – retrieves the closest disaster example and classifies its type.

All notebooks rely on common packages such as `requests`, `pandas`, `numpy`, `matplotlib`, `plotly`, and `scikit-learn`. When an Ollama server is not available, the notebooks use built-in simulation so they still execute.

### Running

Install the optional dependencies and execute a notebook with `jupyter nbconvert`:

```bash
pip install pandas numpy matplotlib plotly scikit-learn pillow requests nbformat nbconvert
jupyter nbconvert --execute notebooks/1_simple_starter.ipynb
```

To use your own Ollama models (e.g. `llama3`, `llava`, or `mxbai-embed-large`), set `simulate = False` inside the notebooks.
