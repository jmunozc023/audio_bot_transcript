from rag.indexer import build_index

index = None

def ask_question(question):
    global index
    if index is None:
        index = build_index()
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return str(response)
