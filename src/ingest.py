"""Pipeline de ingesta: documentos -> chunks con metadatos -> ChromaDB.

Uso:
    python -m src.ingest                       # reindexa todo data/raw
    python -m src.ingest --dry-run             # solo muestra estadisticas de chunking
    python -m src.ingest --chunk-size 800 --chunk-overlap 120 --collection exp_800

Decisiones clave:
  * Chunking *consciente de la estructura legal*: primero se parte por
    ARTICULO (unidad semantica natural de una ley y la unidad que el usuario
    necesita ver citada); solo los articulos que exceden `chunk_size` se
    subdividen con RecursiveCharacterTextSplitter (parrafo -> PARAGRAFO ->
    linea -> frase -> palabra), con solapamiento para no perder el hilo.
  * Cada chunk se embebe con un encabezado ("Ley 769 de 2002 - Art. 131 ...")
    para que el vector capture a que norma pertenece; en la BD se guarda solo
    el texto limpio y el encabezado va en metadatos (`cita`).
  * La indexacion es una reconstruccion completa (borra y recrea la coleccion),
    lo que evita fragmentos obsoletos si cambian los parametros.
"""

import argparse
import dataclasses
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import AppConfig, ConfigError
from .embeddings import GeminiEmbedder
from .retriever import abrir_cliente

EXTENSIONES = {".pdf", ".html", ".htm", ".txt", ".docx"}

# --------------------------------------------------------------------------
# 1. Carga de texto
# --------------------------------------------------------------------------


def _leer_pdf(ruta: Path) -> str:
    from pypdf import PdfReader

    return "\n".join((p.extract_text() or "") for p in PdfReader(str(ruta)).pages)


def _leer_html(ruta: Path) -> str:
    from bs4 import BeautifulSoup

    # Se pasan bytes para que BeautifulSoup detecte el charset del <meta>
    # (los HTML legislativos antiguos suelen venir en windows-1252).
    soup = BeautifulSoup(ruta.read_bytes(), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text("\n")


def _leer_txt(ruta: Path) -> str:
    datos = ruta.read_bytes()
    try:
        return datos.decode("utf-8")
    except UnicodeDecodeError:
        return datos.decode("latin-1")


def _leer_docx(ruta: Path) -> str:
    from docx import Document

    return "\n".join(p.text for p in Document(str(ruta)).paragraphs)


_LECTORES = {
    ".pdf": _leer_pdf,
    ".html": _leer_html,
    ".htm": _leer_html,
    ".txt": _leer_txt,
    ".docx": _leer_docx,
}

# Lineas de "ruido" tipicas de portales legislativos (no son norma).
_RUIDO = re.compile(
    r"^(jurisprudencia|vigencia|legislaci[oó]n anterior|concordancias|notas? de vigencia|"
    r"doctrina concordante|ir al inicio|subir|volver|imprimir|compilaci[oó]n jur[ií]dica.*)\W*$",
    re.IGNORECASE,
)


def limpiar_texto(texto: str) -> str:
    # NFKC descompone ligaduras de PDF ("ﬁ" -> "fi") que romperian la busqueda
    # y convierte el espacio duro (\xa0) en espacio normal.
    texto = unicodedata.normalize("NFKC", texto)
    lineas = []
    for linea in texto.splitlines():
        linea = re.sub(r"[ \t]+", " ", linea).strip()
        if linea and not _RUIDO.match(linea):
            lineas.append(linea)
    return "\n".join(lineas)


def nombre_documento(ruta: Path) -> str:
    """'ley_0769_2002.html' -> 'Ley 769 de 2002'; otros: nombre del archivo legible."""
    m = re.match(r"(ley|decreto|resolucion|resolución)[_\- ]0*(\d+)[_\- ](\d{4})", ruta.stem, re.I)
    if m:
        return f"{m.group(1).capitalize()} {m.group(2)} de {m.group(3)}"
    return re.sub(r"[_\-]+", " ", ruta.stem).strip().capitalize()


# --------------------------------------------------------------------------
# 2. Segmentacion estructural (Titulo / Capitulo / Articulo)
# --------------------------------------------------------------------------

_RE_ARTICULO = re.compile(r"^art[íi]culo\s*(\d+(?:-[A-Za-z0-9]+)?)\s*[º°o]?\s*\.\s*(.*)$", re.I)
_RE_TITULO = re.compile(r"^(T[ÍI]TULO\s+(?:[IVXLCDM]+|\d+|[ÚU]NICO|PRELIMINAR)\b.*)$")
_RE_CAPITULO = re.compile(r"^((?:CAP[ÍI]TULO|SECCI[ÓO]N)\s+(?:[IVXLCDM]+|\d+|[ÚU]NICO)\b.*)$")


@dataclass
class Seccion:
    articulo: str
    titulo: str
    capitulo: str
    texto: str


def segmentar(texto: str) -> list[Seccion]:
    """Divide el texto en articulos, arrastrando el Titulo/Capitulo vigentes."""
    lineas = texto.splitlines()
    titulo = capitulo = ""
    secciones: list[Seccion] = []
    actual: Seccion | None = None

    def encabezado(i: int, m: re.Match) -> str:
        # "CAPITULO I" suele traer su nombre en la linea siguiente.
        base = m.group(1).strip()
        siguiente = lineas[i + 1] if i + 1 < len(lineas) else ""
        if len(base) <= 22 and siguiente and not any(r.match(siguiente) for r in (_RE_ARTICULO, _RE_TITULO, _RE_CAPITULO)):
            return f"{base} {siguiente.strip()}"
        return base

    for i, linea in enumerate(lineas):
        if m := _RE_ARTICULO.match(linea):
            actual = Seccion(m.group(1), titulo, capitulo, linea)
            secciones.append(actual)
        elif m := _RE_TITULO.match(linea):
            titulo, capitulo = encabezado(i, m), ""
        elif m := _RE_CAPITULO.match(linea):
            capitulo = encabezado(i, m)
        elif actual is not None:
            actual.texto += "\n" + linea

    if not secciones:  # documento sin estructura de articulos
        return [Seccion("N/A", "", "", texto)]
    return secciones


# --------------------------------------------------------------------------
# 3. Chunking
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    id: str
    texto: str  # texto limpio (lo que se muestra / se da al LLM)
    texto_embedding: str  # encabezado + texto (lo que se vectoriza)
    metadata: dict


# Encabezado de grupo de multa del Art. 131: "C. Sera sancionado con multa
# equivalente a quince (15) SMLDV ... infracciones:". Los items (C.1, C.2...)
# quedan a miles de caracteres de el; sin repetirlo, un chunk con "C.6. No
# utilizar el cinturon" no dice cuanto vale la multa (iteracion de mejora 1).
_RE_GRUPO_MULTA = re.compile(r"(?m)^[A-F]\.\s+Ser[aá]\s+sancionado[^:]{0,400}:")


def _partes(texto: str, splitter: RecursiveCharacterTextSplitter, config: AppConfig) -> list[str]:
    if len(texto) <= config.chunk_size:
        return [texto.strip()]
    docs = splitter.create_documents([texto])
    if not config.repetir_cabecera_grupos:
        return [d.page_content.strip() for d in docs]
    cabeceras = [(m.start(), " ".join(m.group().split())) for m in _RE_GRUPO_MULTA.finditer(texto)]
    partes = []
    for d in docs:
        inicio = d.metadata["start_index"]
        previas = [c for pos, c in cabeceras if pos < inicio]
        cuerpo = d.page_content.strip()
        # Solo se antepone si el encabezado no viene ya dentro del chunk.
        partes.append(f"(Encabezado del grupo: {previas[-1]})\n{cuerpo}" if previas else cuerpo)
    return partes


def crear_chunks(texto: str, documento: str, archivo: str, config: AppConfig) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,  # posicion del chunk en el articulo (para su encabezado de grupo)
        separators=["\n\n", "\nPARÁGRAFO", "\nPARAGRAFO", "\n", "; ", ". ", " ", ""],
    )
    slug = re.sub(r"\W+", "-", documento.lower()).strip("-")
    chunks: list[Chunk] = []

    for sec in segmentar(texto):
        partes = [p for p in _partes(sec.texto, splitter, config) if len(p) >= 40]
        etiqueta = f"Art. {sec.articulo}" if sec.articulo != "N/A" else "Documento"
        cita = f"{documento}, {etiqueta}"
        contexto = " | ".join(x for x in (sec.titulo, sec.capitulo) if x)
        encabezado = f"[{cita}{' | ' + contexto if contexto else ''}]"

        for j, parte in enumerate(partes):
            chunks.append(
                Chunk(
                    id=f"{slug}-{len(chunks):05d}",
                    texto=parte,
                    texto_embedding=f"{encabezado}\n{parte}",
                    metadata={
                        "documento": documento,
                        "articulo": sec.articulo,
                        "titulo": sec.titulo,
                        "capitulo": sec.capitulo,
                        "archivo": archivo,
                        "cita": cita,
                        "parte": j + 1,
                        "total_partes": len(partes),
                    },
                )
            )
    return chunks


# --------------------------------------------------------------------------
# 4. Indexacion
# --------------------------------------------------------------------------


def cargar_corpus(data_dir: Path, config: AppConfig) -> list[Chunk]:
    archivos = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in EXTENSIONES)
    if not archivos:
        raise FileNotFoundError(
            f"No hay documentos en '{data_dir}'. Coloca ahi PDF/HTML/TXT/DOCX del corpus."
        )
    todos: list[Chunk] = []
    for ruta in archivos:
        texto = limpiar_texto(_LECTORES[ruta.suffix.lower()](ruta))
        chunks = crear_chunks(texto, nombre_documento(ruta), ruta.name, config)
        print(f"  {ruta.name}: {len(texto):,} caracteres -> {len(chunks)} chunks")
        todos.extend(chunks)
    return todos


def indexar(chunks: list[Chunk], config: AppConfig) -> None:
    cliente = abrir_cliente(config)
    try:
        cliente.delete_collection(config.collection)
    except Exception:
        pass  # no existia
    coleccion = cliente.create_collection(
        name=config.collection,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": config.embedding_model,
            "embedding_dim": config.embedding_dim,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        },
    )
    embedder = GeminiEmbedder(config)
    print(f"Generando embeddings ({config.embedding_model}, {config.embedding_dim}d)...")
    vectores = embedder.embed_documentos([c.texto_embedding for c in chunks])
    lote = 500
    for i in range(0, len(chunks), lote):
        sub = slice(i, i + lote)
        coleccion.add(
            ids=[c.id for c in chunks[sub]],
            documents=[c.texto for c in chunks[sub]],
            metadatas=[c.metadata for c in chunks[sub]],
            embeddings=vectores[sub],
        )
    print(f"OK: {coleccion.count()} chunks en '{config.collection}' ({config.chroma_dir}/)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Indexa el corpus normativo en ChromaDB.")
    ap.add_argument("--data-dir")
    ap.add_argument("--chroma-dir")
    ap.add_argument("--collection")
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--chunk-overlap", type=int)
    ap.add_argument("--dry-run", action="store_true", help="solo chunking, sin embeddings")
    args = ap.parse_args()

    try:
        config = AppConfig.from_env(require_api_key=not args.dry_run)
    except ConfigError as e:
        sys.exit(f"[Error de configuracion] {e}")

    cambios = {
        "data_dir": args.data_dir,
        "chroma_dir": args.chroma_dir,
        "collection": args.collection,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    config = dataclasses.replace(config, **{k: v for k, v in cambios.items() if v is not None})
    if config.chunk_overlap >= config.chunk_size:
        sys.exit("chunk_overlap debe ser menor que chunk_size")

    print(f"Corpus: {config.data_dir} | chunk_size={config.chunk_size} overlap={config.chunk_overlap}")
    chunks = cargar_corpus(Path(config.data_dir), config)
    largos = [len(c.texto) for c in chunks]
    print(f"Total: {len(chunks)} chunks | longitud media {sum(largos) // len(largos)} | max {max(largos)}")

    if args.dry_run:
        for c in chunks[:3]:
            print(f"\n--- {c.metadata['cita']} (parte {c.metadata['parte']}/{c.metadata['total_partes']})\n{c.texto[:300]}")
        return
    indexar(chunks, config)


if __name__ == "__main__":
    main()
