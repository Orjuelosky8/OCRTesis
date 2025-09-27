from qdrant_client import QdrantClient, models as qm

q = QdrantClient(url="http://127.0.0.1:6333")

page = q.scroll(
    collection_name="tesis_chunks",
    limit=3,
    with_payload=True,
    with_vectors=False,   # pon True si quieres ver el vector (grande)
)
for pt in page[0]:
    print(pt.id, pt.payload["source_name"], pt.payload["chunk_index"])
    print(pt.payload["text"][:240], "\n")