# Local ChatPDFProcessor with DeepSeek R1

**ChatPDFProcessor** is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interact with them through a chatbot interface. The system uses advanced embedding models and a local vector store for efficient and accurate question-answering.

## Features

- **PDF Upload**: Upload one or multiple PDF documents to enable question-answering across their combined content.
- **RAG Workflow**: Combines retrieval and generation for high-quality responses.
- **Customizable Retrieval**: Adjust the number of retrieved results (`k`) and similarity threshold to fine-tune performance.
- **Memory Management**: Easily clear vector store and retrievers to reset the system.
- **Streamlit Interface**: A user-friendly web application for seamless interaction.

## Problem Statement

The project addresses the challenge of interacting with and querying PDF documents efficiently. It leverages the Retrieval-Augmented Generation (RAG) approach to provide users with accurate responses to questions related to uploaded documents. Users can upload multiple PDFs and receive tailored, relevant answers to their inquiries.

## Proposed Solution

The solution involves a web-based application where users can upload PDFs, and the system uses embeddings to extract document content for efficient and accurate question-answering. The application allows customization of retrieval settings, such as adjusting the number of retrieved results or the similarity threshold for optimal performance.

![img](/asset/rag_local.jpeg)

## Major Tech Stack Explanation

- **Streamlit**: A Python web framework for creating interactive, data-driven web applications quickly. It is used here for the front-end interface that allows users to interact with the chatbot and upload PDFs.
  
- **LangChain**: A framework for building applications powered by language models. It provides tools for chaining together multiple LLMs and allows the integration of various document-processing features such as RAG.

- **Ollama**: A platform for serving language models and embedding models. It's used in this project to run the deep learning models that handle the embedding and question-answering tasks.

- **PyPDF**: A library to work with PDF documents in Python. It is used here to read and extract content from the uploaded PDFs.

- **ChromaDB**: A vector store that helps store and retrieve embeddings for efficient similarity-based retrieval. It’s used here to persist document embeddings and retrieve relevant context during question-answering.


## Installation

Follow the steps below to set up and run the application:

### 1. Clone the Repository

```bash
git clone https://github.com/ajitsingh98/Building-RAG-System-with-Deepseek-R1-Locally.git
cd Building-RAG-System-with-Deepseek-R1-Locally
```

### 2. Create a Virtual Environment

```bash
python3 -m venv rag_pdf
source rag_pdf/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Make sure to include the following packages in your `requirements.txt`:

```
streamlit
langchain
langchain_ollama
langchain_community
streamlit-chat
pypdf
chromadb
```

### 4. Pull Required Models for Ollama

To use the specified embedding and LLM models (`mxbai-embed-large` and `deepseek-r1`), download them via the `ollama` CLI:

```bash
ollama pull mxbai-embed-large
ollama pull deepseek-r1:latest
```

---

## Usage

### 1. Start the Application

Run the Streamlit app:

```bash
streamlit run app.py
```

### 2. Upload Documents

- Navigate to the **Upload a Document** section in the web interface.
- Upload one or multiple PDF files to process their content.
- Each file will be ingested automatically and confirmation messages will show processing time.

### 3. Ask Questions

- Type your question in the chat input box and press Enter.
- Adjust retrieval settings (`k` and similarity threshold) in the **Settings** section for better responses.

### 4. Clear Chat and Reset

- Use the **Clear Chat** button to reset the chat interface.
- Clearing the chat also resets the vector store and retriever.

---

## Project Structure

```
.
├── app.py                  # Streamlit app for the user interface
├── main.py                 # Core RAG logic for PDF ingestion and question-answering
├── requirements.txt        # List of required Python dependencies
├── chroma_db/              # Local persistent vector store (auto-generated)
└── README.md               # Project documentation
```

## Configuration

You can modify the following parameters in `main.py` to suit your needs:

1. **Models**:
   - Default LLM: `deepseek-r1:latest` (7B parameters)
   - Default Embedding: `mxbai-embed-large` (1024 dimensions)
   - Change these in the `ChatPDFProcessor` class constructor or when initializing the class
   - Any Ollama-compatible model can be used by updating the `llm_model` parameter

2. **Chunking Parameters**:
   - `chunk_size=1024` and `chunk_overlap=100`
   - Adjust for larger or smaller document splits

3. **Retrieval Settings**:
   - Adjust `k` (number of retrieved results) and `score_threshold` in `ask()` to control the quality of retrieval.

## Requirements

- **Python**: 3.8+
- **Streamlit**: Web framework for the user interface.
- **Ollama**: For embedding and LLM models.
- **LangChain**: Core framework for RAG.
- **PyPDF**: For PDF document processing.
- **ChromaDB**: Vector store for document embeddings.

## Troubleshooting

### Common Issues

1. **Missing Models**:
   - Ensure you've pulled the required models using `ollama pull`.

2. **Vector Store Errors**:
   - Delete the `chroma_db/` directory if you encounter dimensionality errors:
     ```bash
     rm -rf chroma_db/
     ```

3. **Streamlit Not Launching**:
   - Verify dependencies are installed correctly using `pip install -r requirements.txt`.


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Streamlit](https://github.com/streamlit/streamlit)
- [Ollama](https://ollama.ai/)


### Remarks

This project offers a powerful and easy-to-use system for interacting with PDFs through advanced AI models. Its modular setup and customizable configurations allow for flexible deployment and enhancement, such as adding persistent memory or expanding model support.

### How to Use

1. **Clone the repository** and install the dependencies.
2. **Pull required models** using Ollama.
3. **Start the application** with Streamlit.
4. **Upload PDFs** through the user interface.
5. **Ask questions** and receive answers based on the content of the uploaded PDFs.
6. Adjust retrieval settings and clear chats as needed.

Let me know if you need any further clarifications or edits!