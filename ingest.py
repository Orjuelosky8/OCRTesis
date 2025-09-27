import os, re, io, shutil, hashlib, tempfile, zipfile, datetime, concurrent.futures
from pathlib import Path
from typing import List, Dict, Iterable, Tuple
import uuid
from datetime import datetime, timezone

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

def upsert_chunks(chunks, meta, batch_size: int = 128):
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
            # UUID v5 determinístico: mismo doc+índice => mismo ID
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{base_key}:{i+j}"))

            points.append(
                PointStruct(
                    id=pid,  # <- ahora es un UUID válido
                    vector=v,
                    payload={
                        "text": c,
                        "chunk_index": i + j,
                        **meta,
                        "created_at": now
                    }
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(points)
    return total

    
# ------------ Pipeline por archivo ------------
def process_file(path: Path) -> Tuple[str, int]:
    try:
        suffix = path.suffix.lower()
        if suffix in ARCHIVES:
            # Extrae y procesa recursivamente
            with tempfile.TemporaryDirectory() as td:
                extract_archive(path, Path(td))
                count = 0
                for p in Path(td).rglob("*"):
                    if p.is_file():
                        _, c = process_file(p)
                        count += c
                return (str(path), count)

        if suffix in {".html", ".htm", ".xhtml"}:
            text = html_to_text(path)
        elif suffix in SUPPORTED_EXT:
            text = convert_with_docling(path)
        else:
            # Ignora formatos no soportados
            return (str(path), 0)

        base_meta = {
            "source_path": str(path),
            "source_name": path.name,
            "source_ext": suffix,
            "sha256": file_hash(path),
        }



        # Segmentación adicional por páginas/secciones si el MD trae separadores
        # Aquí usamos chunking uniforme por caracteres
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"[CHUNKS] {path.name}: {len(chunks)}")
        if not chunks:
            return (f"SKIP {path.name} (chunking vacío)", 0)

        insertados = upsert_chunks(chunks, base_meta)
        print(f"[UPSERT] {path.name}: {insertados} puntos")
        return (str(path), len(chunks))

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n[INGEST ERROR] {path}\n{e}\n{tb}\n")
        return (f"ERROR {path}", -1)

def walk_and_process(root: Path):
    files = [p for p in root.rglob("*") if p.is_file()]
    print(f"Archivos detectados: {len(files)}")
    total_chunks = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for fp, n in tqdm(ex.map(process_file, files), total=len(files), desc="Ingesta"):
            if n > 0: total_chunks += n
            elif n == -1: print(fp)
    print(f"Listo. Chunks creados: {total_chunks}")

if __name__ == "__main__":
    print(f"Iniciando ingesta desde {INPUT_DIR}")
    walk_and_process(INPUT_DIR)
