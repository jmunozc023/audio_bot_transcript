from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

def build_index():
    # Load documents from the specified directory
    documents = SimpleDirectoryReader(
        input_files=["data/call.txt"]
    ).load_data()
    
    # Create a VectorStoreIndex from the loaded documents
    index = VectorStoreIndex.from_documents(documents)
    
    return index