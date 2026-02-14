from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from pathlib import Path
from datetime import datetime
from loguru import logger
import tiktoken
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions,TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

class DocumentProcessor:
    def __init__(self):
        
        # Following code will update OCR system for Docling @ 
        #https://docling-project.github.io/docling/examples/full_page_ocr/
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = EasyOcrOptions(
        lang=["en"],  # add languages as needed
        )
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True
        
        )
        self.converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
        )
        
        # Default Docling implementation with RapidOCR
        # self.converter = DocumentConverter()
        self.chunker = HybridChunker()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def process_document(
        self,
        file_path: str,
        file_type: str
    ) -> tuple[list[str], list[dict]]:
        """
        Process document and return chunks with metadata
        Returns: (chunks, metadata_list)
        """
        try:
            # Convert document using Docling
            result = self.converter.convert(file_path)
            doc = result.document
            
            # Apply hybrid chunking
            chunk_iter = self.chunker.chunk(doc)
            
            chunks = []
            metadatas = []
            
            for idx, chunk in enumerate(chunk_iter):
                content = chunk.text
                chunks.append(content)
                
                # Extract metadata
                metadata = self._create_metadata(
                    chunk=chunk,
                    chunk_index=idx,
                    source_file=Path(file_path).name,
                    file_type=file_type,
                    content=content
                )
                metadatas.append(metadata)
            
            logger.info(f"Processed {len(chunks)} chunks from {file_path}")
            return chunks, metadatas
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            raise
    
    def _create_metadata(
        self,
        chunk,
        chunk_index: int,
        source_file: str,
        file_type: str,
        content: str
    ) -> dict:
        """Create metadata dictionary for chunk"""
        
        tokens = len(self.tokenizer.encode(content))
        chars = len(content)
        preview = content[:100] + "..." if len(content) > 100 else content
        
        # Extract keywords (simple approach)
        keywords = self._extract_keywords(content)
        
        now = datetime.utcnow()
        
        metadata = {
            "chunk_id": f"{source_file}_{chunk_index}",
            "source_file": source_file,
            "file_type": file_type,
            "chunk_index": chunk_index,
            "total_chunks": -1,  # Will update after processing all chunks
            "chunk_method": "hybrid",
            "token_count": tokens,
            "char_count": chars,
            "content_preview": preview,
            "keywords": keywords,
            "created_at": now.isoformat(),
            "processed_at": now.isoformat()
        }
        
        # Add Docling-specific metadata if available
        if hasattr(chunk, 'meta'):
            if hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                first_item = chunk.meta.doc_items[0]
                metadata["doc_item_type"] = first_item.label
                if hasattr(first_item, 'prov') and first_item.prov:
                    # ProvenanceItem has page_no attribute (not page)
                    metadata["page_number"] = first_item.prov[0].page_no
        
        return metadata
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> list[str]:
        """Simple keyword extraction"""
        words = text.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [w for w in words if len(w) > 4 and w not in stop_words]
        return list(dict.fromkeys(keywords))[:max_keywords]
    
    def update_total_chunks(self, metadatas: list[dict]) -> list[dict]:
        """Update total_chunks field after processing"""
        total = len(metadatas)
        for metadata in metadatas:
            metadata["total_chunks"] = total
        return metadatas
