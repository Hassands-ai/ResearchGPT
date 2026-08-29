# 🔬 ResearchGPT — AI-Powered Research Assistant

> An AI-powered research assistant for understanding, searching, and interacting with research papers using Large Language Models, semantic search, embeddings, and Retrieval-Augmented Generation (RAG).

---

## 📌 Overview

ResearchGPT is an end-to-end AI research assistant designed to help researchers and students work with research papers more efficiently.

Instead of manually reading large documents and searching through multiple pages, ResearchGPT provides an intelligent interface for processing research papers, creating semantic representations, retrieving relevant information, and generating AI-powered responses.

The project combines:

- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Semantic embeddings
- Vector search
- Research-paper processing
- FastAPI backend
- PostgreSQL database
- Qdrant vector database
- Redis
- MinIO object storage

---

## 🎯 Project Objectives

The main objectives of ResearchGPT are:

1. Process research papers automatically.
2. Extract and prepare document content for AI processing.
3. Convert document content into semantic embeddings.
4. Store embeddings in a vector database.
5. Retrieve the most relevant information for a user query.
6. Provide context-aware AI-generated responses.
7. Reduce the time required to search and understand research papers.
8. Provide a foundation for an intelligent research assistant.

---

## ✨ Key Features

### 📄 Research Paper Processing
Upload and process research documents so their content can be prepared for AI-based retrieval.

### 🔎 Semantic Search
Search for information based on meaning rather than only exact keyword matches.

### 🧠 Embeddings
Research-paper content is transformed into numerical vector representations using embedding models.

### 📚 Retrieval-Augmented Generation

ResearchGPT follows a RAG-based workflow:

```text
Research Paper
      ↓
Document Processing
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
Vector Database
      ↓
User Query
      ↓
Semantic Retrieval
      ↓
Relevant Context
      ↓
LLM
      ↓
AI Response
