# KPoEM

The **KPoEM repository** provides a comprehensive framework for the computational analysis and creative generation of Korean poetry, integrating specialized deep learning models with digital humanities datasets. The workflow begins with automated parsing utilities for **Wikisource** to streamline the curation of modern Korean literary corpora. 

These collected data, alongside the **KPoEM v4** and **KOTE datasets**, are utilized in a robust fine-tuning pipeline for the **KcELECTRA-base model** to achieve high-accuracy emotion classification across 44 distinct labels. The project facilitates seamless deployment by converting PyTorch Lightning checkpoints into Hugging Face formats and implements a sophisticated **Retrieval-Augmented Generation (RAG) system**. 

This system leverages a **FAISS vector store** built with enriched metadata to synthesize new poetry inspired by classic literary nuances.

## File Descriptions
* **`01_dataParsing_Wikisource.ipynb`**
  * **Automated Data Retrieval**: Downloads official Korean Wikisource XML dumps and utilizes `wikiextractor` to convert raw wiki data into structured JSON and Pandas DataFrames.
  * **Structural Parsing**: Implements specialized functions to extract text at both the individual line level and the full-poem level while filtering out unnecessary HTML/Wiki tags.
  * **Targeted Content Extraction**: Features a search-based retrieval system that allows users to isolate specific works by title, author, or index for granular data analysis.
  * **Dataset Export**: Cleans and exports the parsed literary data into standardized `.csv` files, facilitating the creation of custom corpora for digital humanities research.
* **`02_finetuning_kcelectra_kote_kpoem.ipynb`**
  * **Data Integration**: Preprocesses and merges the KOTE (colloquial) and KPoEM v4 (literary) datasets to capture a wide range of Korean emotional expressions.
  * **Sequential Fine-Tuning**: Supports a two-stage training strategy where the model is first fine-tuned on KOTE and subsequently refined on KPoEM.
  * **Automated Optimization**: Utilizes **Optuna** to conduct hyperparameter searches for optimal learning rates, batch sizes, and dropout configurations.
  * **Rigorous Evaluation**: Computes comprehensive performance metrics—including Accuracy, Micro/Macro F1-score, and MCC—to benchmark different training experiments on a common test set.
* **`03_ckpt2hf.ipynb`**
  * **Checkpoint Migration**: Loads the trained model from a PyTorch Lightning `.ckpt` file and extracts the `KcELECTRA` backbone along with the custom classification head.
  * **Format Transformation**: Converts the model weights and configurations into the Hugging Face Transformers standard, saving files such as `config.json` and `model.safetensors`.
  * **Weight Extraction**: Separately exports the state dictionary of the 44-label emotion classifier as `classifier_state.bin` for modular use.
  * **Consistency Validation**: Runs a comparison test between the original checkpoint and the converted model to ensure the output logits remain identical.
* **`04_KPoEM_vectorstore.ipynb`**
  * **Metadata Enrichment**: Merges line-level and poem-level datasets while applying a specific min-max scaling (threshold 0.2) to emotion annotations for high-fidelity metadata.
  * **Custom Embedding Implementation**: Defines a specialized `KcELECTRAEmbeddings` class using the latest `KcELECTRA-base` model to represent Korean poetic nuances in a high-dimensional vector space.
  * **Vector Database Construction**: Transforms the enriched KPoEM corpus into LangChain-compatible `Document` objects and builds a searchable index using **FAISS** (Facebook AI Similarity Search).
  * **Local Persistence**: Serializes the trained vector store to local storage, enabling efficient and fast similarity searches for downstream RAG applications and creative writing tools.
* **`05_KPoEM_poetry_generator.ipynb`**
  * **Emotion-Driven Synthesis**: Integrates the `KPoEM_Classifier` to analyze the emotional nuance of input text and use the results as the primary creative constraints for a Large Language Model (LLM).
  * **RAG-Enhanced Generation**: Implements a Retrieval-Augmented Generation (RAG) pipeline that retrieves contextually and emotionally similar verses from a vector database to guide the LLM's style.
  * **LLM Pipeline Wrapper**: Utilizes LangChain to wrap advanced Korean models (like `Midm-2.0`) and specialized prompt templates to ensure the output maintains a consistent poetic tone.
  * **Heuristic Context Filtering**: Filters retrieved candidates from the vector store based on their emotional metadata overlap to provide the most relevant "poetic triggers" for the generative process.
* **`05-1_[colab]KPoEM_poetry_generator.ipynb`**
  *  This is the version of the `KPoEM_poetry_generator.ipynb` file specifically configured for execution in Google Colab.

## Related Resources
* **KPoEM (Korean Poetry Emotion Mapping) Dataset**​
  * [https://doi.org/10.5281/zenodo.15598092](https://doi.org/10.5281/zenodo.15598092)
* **KPoEM (Korean Poetry Emotion Mapping) Model**
  * [AKS-DHLAB/KPoEM](https://huggingface.co/AKS-DHLAB/KPoEM)
    * [AKS-DHLAB/KcELECTRA_KOTEOnly](https://huggingface.co/AKS-DHLAB/KcELECTRA_KOTEOnly)
    * [AKS-DHLAB/KcELECTRA_KPoEMOnly](https://huggingface.co/AKS-DHLAB/KcELECTRA_KPoEMOnly) 

## Citations
* LIM, I., Ji, H., & Kim, B. (2026). KPoEM: A Human-Annotated Dataset for Emotion Classification and RAG-Based Poetry Generation in Korean Modern Poetry. *The Review of Korean Studies, 29*(1), 161-206.
  * [10.25024/review.2026.29.1.161](https://doi.org/10.25024/review.2026.29.1.161)
