import streamlit as st ## type: ignore

from PyPDF2 import PdfReader ## type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter ## type: ignore
#langchain-community=-0.0.36

from langchain.vectorstores import Qdrant # type: ignore
from qdrant_client import QdrantClient # type: ignore

from langchain.chains.question_answering import load_qa_chain # type: ignore
from langchain.prompts import PromptTemplate # type: ignore

# Import for Ollama embeddings
from langchain_community.llms import Ollama# type: ignore
from langchain_community.embeddings.ollama import OllamaEmbeddings # type: ignore

import qdrant_client # type: ignore

#import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs: 
        pdf_reader = PdfReader(pdf) 
        for page in pdf_reader.pages: 
            text += page.extract_text()
        return text

def get_text_chunks (text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OllamaEmbeddings(model='llama2', temperature=1)

    qdrant_store = Qdrant.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        url=[REPLACE WITH YOUR CLUSTER URL],
        api_key=[REPLACE WITH YOUR API_KEY], #replace with url, apikey and collection_name
        collection_name=[your collection_name]
)
    
def get_conversational_chain(): 
    # Read the prompt template from a file 
    with open("prompt_template.txt", "r") as f: 
        prompt_template = f.read().strip()

    model = Ollama(model="llama2", temperature=1)

    # Define the prompt template with input variables 'context' and 'question' 
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"]) 
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input():
    embeddings = OllamaEmbeddings(model='llama2', temperature=1)

    client = qdrant_client.QdrantClient(
            url=[REPLACE WITH YOUR CLUSTER URL], 
            api_key=[REPLACE WITH YOUR API_KEY]
    ) # For Qdrant Cloud, None for local instance

    doc_store= Qdrant(
        client=client, collection_name=[your collection_name], 
        embeddings=embeddings,
    )

    query='Respond with only the name (give contact & email) of the top one candidate for each job title(like AI/ML Engineer or'

    # Perform similarity search on the vector store with the query
    docs = doc_store.similarity_search(query) 

    #Get the conversational QA chain
    chain = get_conversational_chain()

    #Generate the response using the QA chain
    response = chain(
        {'input_documents':docs, "question":query},
        return_only_outputs=True
    )

    #Display the response
    print(response)
    st.write('RESPONSE:-')
    st.write('Reply: ',response['output_text'])

def main():
    st.set_page_config("Chat PDF")
    st.header("Langchain Similarity Search")

    # user_question = st.text_input("Ask a Question from the PDF Files")
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload Job description", accept_multiple_files=True)
        pdf_docs2 = st.file_uploader("Upload your resume", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."): 
                # Process the uploaded PDF files
                if pdf_docs and pdf_docs2:
                    
                    raw_text = get_pdf_text(pdf_docs) + get_pdf_text(pdf_docs2)
                    text_chunks = get_text_chunks (raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                else:
                    print('Upload Job description and resume first!!')
    user_input()

if __name__=='__main__':
    main()
