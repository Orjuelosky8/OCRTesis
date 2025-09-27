from qdrant_client import QdrantClient, models as qm
import json

q = QdrantClient(url="http://127.0.0.1:6333")

next_page = None
with open("export_tesis_chunks.ndjson", "w", encoding="utf-8") as f:
    while True:
        points, next_page = q.scroll(
            collection_name="tesis_chunks",
            limit=500,
            with_payload=True,
            with_vectors=False,
            offset=next_page
        )
        for p in points:
            f.write(json.dumps({
                "id": p.id,
                "payload": p.payload
            }, ensure_ascii=False) + "\n")
        if next_page is None:
            break

print("Archivo generado: export_tesis_chunks.ndjson")
