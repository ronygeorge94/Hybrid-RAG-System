# Hybrid-RAG System: Production Engine

A high-performance **Retrieval-Augmented Generation** engine designed for technical document analysis. This system moves beyond "Naive RAG" by implementing a sophisticated hybrid search layer that ensures high precision for technical terminology and deep semantic understanding for complex queries.

## Technical Architecture

The system follows a **Modular Pipeline Architecture**, ensuring that each stage from ingestion to generation is decoupled and independently scalable.

### 1. The Hybrid Retrieval Strategy
Unlike standard RAG systems that rely solely on semantic embeddings, this engine uses a **Reciprocal Rank Fusion (RRF)** approach (via LangChain's Ensemble Retriever):
* **Dense Retrieval (FAISS):** Captures semantic meaning and conceptual relationships using OpenAI `text-embedding-3-small`.
* **Sparse Retrieval (BM25):** Acts as a keyword "anchor," ensuring that specific technical IDs, function names, or unique jargon are not lost in the vector space.
* **The Result:** Superior context accuracy, especially for IT documentation where specific error codes or variable names are critical.

### 2. Intelligent Document Ingestion
The pipeline doesn't just read text; it preserves **Contextual Integrity**:
* **Metadata Propagation:** Every chunk maintains a link to its parent file (`source`) and its location (`page_number`).
* **Recursive Character Splitting:** Chunks are created based on semantic boundaries (paragraphs/sentences) rather than arbitrary character counts to prevent "context shearing."

### 3. Traceability & Citation Engine
To solve the "Black Box" problem of LLMs, this system implements a strict citation protocol:
* **Source Injection:** Relevant filenames are injected into the prompt context.
* **In-Text Attribution:** The model is instructed to cite findings using `[n]` notation.
* **Source Verification:** The final output includes a dedicated "Sources Used" section, allowing users to verify claims against the original documents.

## Detailed Workflow

1.  **Bootstrapping:** `main.py` initializes the `AppConfig` via Pydantic, validating that all paths and model temperatures are within safe production limits.
2.  **Indexing:** The system checks for an existing `index.faiss`. If missing, it triggers a batch ingestion process that converts raw PDFs/TXTs into high-dimensional vectors.
3.  **The Chain (LCEL):** We utilize **LangChain Expression Language (LCEL)** to create a stateless, high-speed pipeline:
    `Search -> Context Formatting -> Prompt Application -> LLM Generation -> Output Parsing`

## Advanced Configuration Parameters

Defined in `configs/config.yaml`:
* **Chunk Size (1000):** Optimized for the context window of GPT-4o while maintaining enough surrounding text for meaning.
* **Overlap (100):** Ensures continuity between chunks so no information is lost at the "seams" of a split.
* **Temperature (0.0):** Set to zero to minimize "creative" hallucinations and maximize factual consistency based on retrieved documents."# Hybrid-RAG-System" 
