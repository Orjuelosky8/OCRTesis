# Ingesta de documentos a Qdrant con Docling

## Requisitos
- Docker Desktop (Windows)
- (Opcional si tienes RAR): Instala **WinRAR** o **7-Zip** y deja `unrar.exe` / `7z.exe` en el PATH. El módulo `rarfile` usa uno de esos backends para extraer .rar.

## Pasos

1. Clona o copia esta carpeta `tesis_ingesta` y edita `.env` si es necesario.
2. Verifica que la carpeta con tus archivos esté en `D:\LicitacionesTesis`.  
   Si no, cambiae el volumen en `docker-compose.yml` (línea `- /d/LicitacionesTesis:/data:ro`).
3. Inicia servicios y la ingesta:

```bash
docker compose up --pull always --build
