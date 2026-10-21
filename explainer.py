
# Import Libraries

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
)
from youtube_transcript_api._errors import IpBlocked
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


# Step 01: Indexing

# 1A: Document Ingestion

video_id = "Gfr50f6ZBvo"

try:
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id)

    transcript = " ".join(
        snippet.text for snippet in transcript_list
    )
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")


# 1b: Text Splitting

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

print(len(chunks))

print(chunks[100])


# 1c: Embedding Generation and Vector Store

embeddings = HuggingFaceEndpointEmbeddings(model="BAAI/bge-small-en-v1.5")
vector_store = FAISS.from_documents(chunks, embeddings)

print(vector_store.index_to_docstore_id)

print(vector_store.get_by_ids(['2436bdb8-3f5f-49c6-8915-0c654c888700']))


# Step 02: Retrieval

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

print(retriever)

retriever.invoke('What is deepmind')


# Step 03: Augmentation

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

print(retrieved_docs)


context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
print(context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})
print(final_prompt)


# Step 04: Generation

answer = llm.invoke(final_prompt)
print(answer.content)


# Building a chain

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text


parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})


parallel_chain.invoke('who is Demis')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

main_chain.invoke('Can you summarize the video')