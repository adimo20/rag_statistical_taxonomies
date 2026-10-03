import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

class VectorStore:
    """
    Interface for filling and querying the chromaDB (persistent) client.

    Parameters:
        collection_name (str) - name of the collection to create/load
        model_name (str) - sentence-transformers model used to embed documents
            and queries. Both sides go through the same model, so a collection
            is only meaningful together with the model that built it.
        chromadb_path (str) - path of the persistent client to load or create
    """

    def __init__(self, collection_name: str, model_name: str, chromadb_path: str) -> None:
        self.collection_name = collection_name
        self.model_name = model_name
        self.chromadb_path = chromadb_path
        self.chroma_client = chromadb.PersistentClient(path=chromadb_path)
        print("Setting up embedding function and collection!")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=SentenceTransformerEmbeddingFunction(model_name), # type: ignore
            configuration={"hnsw": {"space": "cosine"}}, # applying hierarchical small world navigation index for faster similarity search,
            # for just a handfull of description, like in this research inference time ins't a problem, but when embedding a bigger 
            # knowledge base with 100k upwards labelled examples this has a big impact but. Due to the index, filling the KB can take long
            # --> time to fill the knowledge base progressivly increases with number of examples as, the index has to be recomputed, arranged with
            #     each batch. 
        )

    def create_collection(
        self,
        ids:list[str],
        documents:list[str],
        metadatas:list[dict],
        batch_size:int=512
    )->None:
        # Batching only mattern with many examples, if running on a smaller machine 512 may be too high.
        for i in tqdm(range(0, len(ids), batch_size)):
            self.collection.add(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size], # type: ignore
            )
        print("Sucessfully indexed classification system")
