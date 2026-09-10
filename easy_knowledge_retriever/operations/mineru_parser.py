import re
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path

from easy_knowledge_retriever.constants import DEFAULT_MINERU_TIMEOUT

_INLINE_SUP = re.compile(r"\$\s*\^\s*\{\s*([^{}$]*?)\s*\}\s*\$")


def clean_extracted_text(text: str) -> str:
    """Strip layout artefacts that break exact citations: amendment markers (▌) and the
    inline LaTeX superscripts MinerU emits for reference calls (``$^ { 16 }$`` -> ``16``)."""
    text = _INLINE_SUP.sub(lambda m: m.group(1).replace(" ", ""), (text or "").replace("▌", ""))
    return re.sub(r"[ \t]{2,}", " ", text)


class MineruParser:
    """
    Parser for PDF documents using Mineru (Magic-PDF) CLI.
    """

    def __init__(self, output_dir: Optional[str] = None, timeout: Optional[int] = None):
        """
        Initialize the parser.
        
        Args:
            output_dir: Optional directory to store parsed outputs. 
                        If None, a temporary directory is used.
            timeout: Max seconds the MinerU subprocess may run before the parse
                     is aborted. Defaults to EKR_MINERU_TIMEOUT (1h).
        """
        self.output_dir = output_dir
        self.timeout = timeout if timeout is not None else DEFAULT_MINERU_TIMEOUT

    def parse(self, file_path: str, start_page: Optional[int] = None, end_page: Optional[int] = None) -> Dict[str, Any]:
        """
        Parse a PDF file and return structured content.
        
        Args:
            file_path: Absolute path to the PDF file.
            start_page: Optional start page index (0-based).
            end_page: Optional end page index (0-based) or count.
            
        Returns:
            Dict containing content, pages, etc.
        """
        file_path_obj = Path(file_path).resolve()
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine output directory
        use_temp_dir = self.output_dir is None
        working_dir = Path(self.output_dir) if self.output_dir else Path(tempfile.mkdtemp(prefix="mineru_"))
        
        try:
            # Construct command to run mineru_tool.py
            current_dir = Path(__file__).parent
            mineru_tool = current_dir / "mineru_tool.py"
            
            # Detect device
            import torch
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            
            print(f"Using device: {device}")

            cmd = [
                sys.executable,
                str(mineru_tool),
                "-p", str(file_path_obj),
                "-o", str(working_dir),
                "--device", device
            ]
            
            if start_page is not None:
                cmd.extend(["-s", str(start_page)])
            if end_page is not None:
                cmd.extend(["-e", str(end_page)])

            
            print(f"Running Mineru: {' '.join(cmd)}")
            # A hung MinerU run must fail the ingestion, not hang it forever.
            # Tune with EKR_MINERU_TIMEOUT (seconds).
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Mineru timed out after {self.timeout}s on {file_path_obj}. "
                    f"Raise EKR_MINERU_TIMEOUT if the document is genuinely this large."
                ) from exc

            if result.returncode != 0:
                raise RuntimeError(f"Mineru failed: {result.stderr}\nStdout: {result.stdout}")
                
            # Parse output
            # Output structure: working_dir / file_stem / 'auto' / file_stem_content_list.json
            file_stem = file_path_obj.stem
            # Mineru might handle spaces in filenames by preserving them
            output_subdir = working_dir / file_stem / "auto"
            content_list_path = output_subdir / f"{file_stem}_content_list.json"
            
            if not content_list_path.exists():
                # Fallback: check if directory name replaces spaces or other chars?
                # But based on our test, it kept "2024 PSE" as directory name.
                # Let's try to find the json file if exact match fails
                json_files = list(working_dir.glob(f"**/*_content_list.json"))
                if not json_files:
                    # Include stdout/stderr in error message for debugging
                    raise FileNotFoundError(f"Output JSON not found in {working_dir}.\nMineru Stdout: {result.stdout}\nMineru Stderr: {result.stderr}")
                content_list_path = json_files[0]
                output_subdir = content_list_path.parent

            with open(content_list_path, "r", encoding="utf-8") as f:
                content_list = json.load(f)
                
            return self._process_content_list(content_list, output_subdir)

        finally:
            if use_temp_dir and working_dir.exists():
                shutil.rmtree(working_dir, ignore_errors=True)

    def _process_content_list(self, content_list: List[Dict], output_subdir: Path) -> Dict[str, Any]:
        """Process the raw list from Mineru into a structured format."""
        
        full_text = []
        pages = {}
        images = []
        
        for item in content_list:
            page_idx = item.get("page_idx", 0)
            item_type = item.get("type")
            bbox = item.get("bbox")
            
            if page_idx not in pages:
                pages[page_idx] = {"content": "", "images": [], "page_number": page_idx + 1}
            
            if item_type == "text":
                text = clean_extracted_text(item.get("text", ""))
                if text:
                    full_text.append(text)
                    pages[page_idx]["content"] += text + "\n"
            
            elif item_type == "image":
                img_rel_path = item.get("img_path")
                if img_rel_path:
                    # Resolve absolute path or keep relative if we are copying?
                    # Since we delete temp dir, we should probably read the image bytes 
                    # OR copy it to a persistent storage if needed.
                    # For now, we will store the absolute path BUT it will be invalid if temp dir is deleted.
                    # TODO: Integrate with BlobStorage or similar if we want to keep images.
                    # For this implementation, we just list it.
                    abs_path = output_subdir / img_rel_path
                    if abs_path.exists():
                         # Check if we should persist it? 
                         # For now, let's just keep the path but warn it might disappear if temp.
                         # In a real ingest, we should move these images to `self.working_dir/images`
                         pass
                    pages[page_idx]["images"].append(str(abs_path))
                    images.append(str(abs_path))
            
            elif item_type == "table":
                # Handle table text/html
                table_html = item.get("table_body", "")
                if table_html:
                    full_text.append(table_html)
                    pages[page_idx]["content"] += table_html + "\n"
                    
                img_rel_path = item.get("img_path")
                if img_rel_path:
                    abs_path = output_subdir / img_rel_path
                    pages[page_idx]["images"].append(str(abs_path))
        
        # Sort pages by index
        sorted_pages = [pages[k] for k in sorted(pages.keys())]
        
        return {
            "content": "\n".join(full_text),
            "pages": sorted_pages,
            "images": images
        }
