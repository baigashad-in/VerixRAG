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
    def load_file(self, file_path: str) -> Document:
        """Loads a file and returns a Document object."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"No file at {file_path}")
        
        # Read the raw text
        content = path.read_text(encoding="utf-8")

        # Attach metadata - this travels with every chunk
        metadata = {
            "source": str(path.absolute()),
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
