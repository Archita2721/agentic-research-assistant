from langchain_text_splitters import RecursiveCharacterTextSplitter

from constants import TEXT_SPLITTER_CHUNK_OVERLAP, TEXT_SPLITTER_CHUNK_SIZE

def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
        chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)
