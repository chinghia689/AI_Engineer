import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph


#load api
def load_config():
    load_dotenv()
    API_KEY=os.getenv('GOOGLE_API_KEY')
    if not API_KEY:
        raise ValueError("No Key")


class State(TypedDict,total=False):
    question:str
    context:str 
    answer:str

def built_app():

    load_config()


    loader=PyPDFLoader('/home/chinghia/AI_Engineer/RAG_PDF/PDF.pdf')
    docs=loader.load()

    text_split=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    split=text_split.split_documents(docs)

    embeddings=HuggingFaceEmbeddings(model_name='keepitreal/vietnamese-sbert',model_kwargs={'device': 'cuda'})

    vector=Chroma.from_documents(documents=split,embedding=embeddings)

    retriever=vector.as_retriever(search_kwargs={'k':3})


    llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.3)


    def retriever_node(state : State):
        retriever_docs=retriever.invoke(state['question'])
        context_text='\n\n'.join(doc.page_content for doc in retriever_docs)
        return {'context' : context_text}


    def generate_node(state : State):
        template='Context:{context}' \
        ' Question:{question}'
        prompt=ChatPromptTemplate.from_template(template)
        chain=prompt | llm | StrOutputParser()
        response=chain.invoke(state)
        return {"answer": response}


    workflow=StateGraph(State)
    workflow.add_node('retriever',retriever_node)
    workflow.add_node('generate',generate_node)

    workflow.add_edge(START,'retriever')
    workflow.add_edge('retriever','generate')
    workflow.add_edge('generate',END)
    return workflow.compile()
if __name__=='__main__':
    app = built_app()
    print('Nhap vao cau hoi')
    text=input()
    result = app.invoke({"question": text})
    print(result["answer"])
