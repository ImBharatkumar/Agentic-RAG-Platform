import logging
from pathlib import Path


from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

_log = logging.getLogger(__name__)


def docling_ocr(input_path: str):
    input_paths = [Path(input_path)]

    doc_converter = DocumentConverter(  # all of the below is optional, has internal defaults.
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.CSV,
            InputFormat.MD,
        ],  # whitelist formats, non-matching files are ignored.
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=PyPdfiumDocumentBackend,
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline,  # or set a backend, e.g., MsWordDocumentBackend
                # If you change the backend, remember to import it, e.g.:
                #   from docling.backend.msword_backend import MsWordDocumentBackend
            ),
        },
    )

    conv_results = doc_converter.convert_all(input_paths)

    for res in conv_results:
        return res.document.export_to_markdown()  # Return Markdown text
    return ""
