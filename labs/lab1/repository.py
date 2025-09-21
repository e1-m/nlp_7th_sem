from abc import ABC, abstractmethod
import pandas as pd
from pydantic import BaseModel


class Document(BaseModel):
    id: int
    description: str
    display_name: str
    category: str


class DocumentRepository(ABC):
    @abstractmethod
    def get_all_documents(self) -> list[Document]:
        pass


class CSVDocumentRepository(DocumentRepository):
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path, sep='|')

    def get_all_documents(self) -> list[Document]:
        documents = []
        for _, row in self.df.iterrows():
            document = Document(
                id=row['id'],
                description=row['description'],
                display_name=row['display name'],
                category=row['category']
            )
            documents.append(document)
        return documents


if __name__ == "__main__":
    repo = CSVDocumentRepository('data/corpus.csv')
    docs = repo.get_all_documents()
    for doc in docs[:5]:
        print(doc)
