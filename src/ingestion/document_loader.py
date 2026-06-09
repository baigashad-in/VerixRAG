import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Document:
    """A loaded document with its content and metadata.
    
    Metadata is used to cite chunks. It's sources such as which file, which page, etc. It is critical for citations.
    """

    content: str
    metadata: dict # source file, page number, title, etc.

class DocumentLoader:
    """Loads documents from various file formats into plain text.
    
    This supports PDF, DOCX, HTML, Markdown, etc. in the porduction.
    """
    def __init__(self, allowed_directory: str = "./documents"):
        self.allowed_dir = Path(allowed_directory).resolve()

    def load_file(self, file_path: str) -> Document:
        """Loads a file and returns a Document object."""
        path = Path(file_path).resolve()

        # Prevent path traversal — file must be inside allowed directory
        if not str(path).startswith(str(self.allowed_dir)):
            raise PermissionError(
                f"Access denied: {file_path} is outside "
                f"allowed directory {self.allowed_dir}"
            )

        if not path.exists():
            raise FileNotFoundError(f"No file at {file_path}")


        # Limit file size to prevent memory exhaustion
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size} bytes "
                f"(max {MAX_FILE_SIZE})"
            )
        
        # Read the raw text
        content = path.read_text(encoding="utf-8")

        # Attach metadata - this travels with every chunk
        metadata = {
            "source": path.name,
            "filename": path.name,
            "file_type": path.suffix,
            "char_count": len(content),
        }

        return Document(content=content, metadata=metadata)
    

    def load_directory(self, dir_path: str,
                       extensions: list[str] = None) -> list[Document]:
        """Load all matching files from a directory."""
        extensions = extensions or [".txt", ".md"]
        docs = []

        for file_path in Path(dir_path).rglob("*"):
            if file_path.suffix in extensions:
                docs.append(self.load_file(str(file_path)))

        print(f"Loaded {len(docs)} documents")
        return docs
