import os, re, io, shutil, hashlib, tempfile, zipfile, datetime, concurrent.futures
from pathlib import Path
from typing import List, Dict, Iterable, Tuple
import uuid
from datetime import datetime, timezone
from collections import defaultdict

from docling.document_converter import DocumentConverter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import traceback

import rarfile  # requiere unrar.exe/7z en PATH para *.rar
from bs4 import BeautifulSoup

# ------------ Config ------------
INPUT_DIR = Path(os.getenv("INPUT_DIR", "/data"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tesis_chunks")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "jinaai/jina-embeddings-v2-base-es")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

SUPPORTED_EXT = {
    ".pdf", ".docx", ".xlsx", ".pptx", ".md", ".adoc",
    ".html", ".htm", ".xhtml", ".csv",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
    ".vtt"
}
ARCHIVES = {".zip", ".rar"}

# Extrae lic_id y tag del nombre
NAME_RX = re.compile(r'^(\d+)_([A-Za-z0-9\-\.\s]+)')

def parse_name(filename: str):
    m = NAME_RX.match(filename)
    if not m:
        return None, None
    lic_id = m.group(1)
    tag = m.group(2).strip().lower()
    return lic_id, tag

# Orden recomendado por “tipo” para recorrer en una licitación
PRIORITY = [
    "aviso", "convocatoria", "pliego", "pliegos", "anexo",
    "adenda", "aclaracion", "aclaraciones",
    "oferta", "propuesta", "evaluacion", "informe",
    "acta", "audiencia", "resolucion",
    "adjudicacion", "contrato", "interventoria", "poliza",
    "permiso", "certificado", "otros"
]

def order_index(tag: str, name: str):
    t = (tag or "") + " " + name.lower()
    for i, key in enumerate(PRIORITY):
        if key in t:
            return i
    return len(PRIORITY)  # al final si no matchea

# ------------ Utils ------------
def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    """Chunk básico por caracteres, con solapamiento. Respeta límites de palabra."""
    if not text: return []
    words = text.split()
    chunks, cur = [], []
    cur_len = 0
    for w in words:
        add = (len(w) + 1)
        if cur_len + add > size and cur:
            chunks.append(" ".join(cur))
            # overlap
            if overlap > 0:
                back = []
                l = 0
                for ww in reversed(cur):
                    back.append(ww)
                    l += len(ww) + 1
                    if l >= overlap: break
                cur = list(reversed(back))
                cur_len = sum(len(x) + 1 for x in cur)
            else:
                cur, cur_len = [], 0
        cur.append(w); cur_len += add
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def extract_archive(src: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src, "r") as z:
            z.extractall(dest_dir)
    elif src.suffix.lower() == ".rar":
        # Requiere unrar/7z instalado y accesible
        with rarfile.RarFile(src) as r:
            r.extractall(dest_dir)

def html_to_text(path: Path) -> str:
    with open(path, "rb") as f:
        soup = BeautifulSoup(f, "lxml")
    return clean_text(soup.get_text("\n"))

# ------------ Docling ------------
# Docling soporta multi-formato (PDF, DOCX, XLSX, HTML, imágenes con OCR, etc.)
# reference: https://github.com/docling-project/docling (install/usage) y supported formats.
converter = DocumentConverter()

def convert_with_docling(path: Path) -> str:
    """Convierte un archivo soportado a texto markdown con Docling."""
    result = converter.convert(str(path))
    # Exportamos a Markdown unificado (conserva encabezados/tablas razonablemente)
    md = result.document.export_to_markdown()
    return clean_text(md)

# ------------ Embeddings / Qdrant ------------
print("Cargando modelo de embeddings:", EMB_MODEL_NAME)
embedder = SentenceTransformer(EMB_MODEL_NAME)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
vector_dim = embedder.get_sentence_embedding_dimension()

def ensure_collection():
    cols = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in cols:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
ensure_collection()

def upsert_chunks(chunks, meta, lic_chunk_start: int = 0, batch_size: int = 128):
    if not chunks:
        return 0
    total = 0
    base_key = meta.get("sha256") or hashlib.sha256(meta["source_name"].encode("utf-8")).hexdigest()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        vecs = embedder.encode(batch, normalize_embeddings=True).tolist()
        now = datetime.now(timezone.utc).isoformat()

        points = []
        for j, (c, v) in enumerate(zip(batch, vecs)):
            doc_chunk_index = i + j
            lic_chunk_index = lic_chunk_start + doc_chunk_index
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{meta['lic_id']}|{meta['source_name']}|{doc_chunk_index}"))

            points.append(PointStruct(
                id=pid,
                vector=v,
                payload={
                    "text": c,
                    "doc_chunk_index": doc_chunk_index,
                    "lic_chunk_index": lic_chunk_index,
                    **meta,
                    "created_at": now
                }
            ))
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(points)
    return total
    
# ------------ Pipeline por archivo ------------
def process_file_grouped(path: Path, lic_id: str, doc_tag: str | None, doc_order: int | None, lic_chunk_start: int) -> int:
    try:
        suffix = path.suffix.lower()

        # (igual que antes: archives, html_to_text, convert_with_docling...)
        if suffix in ARCHIVES:
            count = 0
            with tempfile.TemporaryDirectory() as td:
                extract_archive(path, Path(td))
                for p in Path(td).rglob("*"):
                    if p.is_file():
                        # Para archivos extraídos, no conocemos orden; usa doc_order incremental local si lo deseas
                        count += process_file_grouped(p, lic_id, f"{doc_tag or 'archivo'}-zip", doc_order, lic_chunk_start + count)
            return count

        if suffix in {".html", ".htm", ".xhtml"}:
            text = html_to_text(path)
        elif suffix in SUPPORTED_EXT:
            text = convert_with_docling(path)
        else:
            return 0

        if not text or len(text.strip()) < 20:
            print(f"SKIP {path.name} (sin texto)")
            return 0

        base_meta = {
            "source_path": str(path),
            "source_name": path.name,
            "source_ext": suffix,
            "sha256": file_hash(path),
            "lic_id": lic_id,
            "doc_tag": doc_tag,
            "doc_order": doc_order,
        }

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"[CHUNKS] {lic_id} | {path.name}: {len(chunks)} (start={lic_chunk_start})")
        if not chunks:
            return 0

        inserted = upsert_chunks(chunks, base_meta, lic_chunk_start=lic_chunk_start)
        print(f"[UPSERT] {lic_id} | {path.name}: {inserted} puntos")
        return len(chunks)

    except Exception as e:
        import traceback
        print(f"\n[INGEST ERROR] {path}\n{e}\n{traceback.format_exc()}\n")
        return 0


def walk_and_process_grouped(root: Path):
    files = [p for p in root.rglob("*") if p.is_file()]
    print(f"Archivos detectados: {len(files)}")

    groups = defaultdict(list)
    for p in files:
        lic_id, tag = parse_name(p.name)
        if lic_id:
            groups[lic_id].append((p, tag))
        else:
            # opcional: procesa los que no matchean fuera de grupos
            groups["_sin_lic_"].append((p, None))

    total_chunks = 0
    for lic_id in sorted(k for k in groups.keys() if k != "_sin_lic_"):
        items = groups[lic_id]
        items.sort(key=lambda it: (order_index(it[1], it[0].name), it[0].name))
        print(f"\n=== Licitación {lic_id} — {len(items)} archivos ===")
        lic_chunk_cursor = 0

        for order, (path, tag) in enumerate(items):
            n_chunks = process_file_grouped(path, lic_id=lic_id, doc_tag=tag, doc_order=order, lic_chunk_start=lic_chunk_cursor)
            lic_chunk_cursor += n_chunks
            total_chunks += n_chunks

    # opcional: procesa los que no cumplen patrón
    if "_sin_lic_" in groups:
        print(f"\n=== Archivos sin patrón — {len(groups['_sin_lic_'])} ===")
        for (path, _) in groups["_sin_lic_"]:
            total_chunks += process_file_grouped(path, lic_id="__without_id__", doc_tag=None, doc_order=None, lic_chunk_start=0)

    print(f"\nListo. Chunks creados: {total_chunks}")


if __name__ == "__main__":
    print(f"Iniciando ingesta desde {INPUT_DIR}")
    walk_and_process_grouped(INPUT_DIR)
