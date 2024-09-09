# LLM HR replacement Chatbot
(users can change their local model with ease)
(Project is Prompt versatile)


This project uses Langchain, Qdrant, Ollama, and Streamlit to create an AI system that compares resumes and job descriptions to suggest the best candidate for a given job. The system utilizes vector similarity search on a set of resumes and job descriptions, providing a ranked output of candidates based on technical skills and experience.

## Note:
This project utilizes Ollama and Ollama Embeddings with Qdrant API , used LLm model(llama2).

### Features
- PDF Processing: Extracts and processes text from uploaded PDFs (job descriptions and resumes).
- Text Chunking: Breaks down long documents into manageable text chunks.
- Vector Search: Utilizes Qdrant's vector search to find relevant resume sections matching job descriptions.
- Conversational QA: Generates responses using Langchain's QA chain powered by the Ollama model.
- Streamlit UI: Simple interface allowing users to upload documents and receive AI-driven candidate suggestions.

### Requirements
Ensure the following dependencies are installed (you can also refer to req.txt for the list):
```
streamlit
PyPDF2
langchain
langchain-community==0.0.36
qdrant-client
langchain_google_genai
```

## Customizing the Prompt
The prompt used to rank candidates is stored in the prompt_template.txt file. You can modify the template to adjust how the AI evaluates resumes. The current template prioritizes technical skills and experience, while ignoring key skills.


