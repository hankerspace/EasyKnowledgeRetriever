from __future__ import annotations
from typing import Any
from easy_knowledge_retriever.utils.logger import logger
from easy_knowledge_retriever.utils.tokenizer import Tokenizer
from easy_knowledge_retriever.kg.exceptions import ChunkTokenLimitExceededError

def _page_lookup(tokenizer: Tokenizer, pages: list[dict[str, Any]] | None):
    """Map a token offset in the document to the page it falls on.

    ponytail: per-page token counts, so an offset can drift by a few tokens at page
    joins (sum of encode(page) is not exactly encode(document)).
    """
    spans, pos = [], 0
    for page in pages or []:
        pos += len(tokenizer.encode(page.get("content", "")))
        spans.append((pos, page.get("page_number")))

    def page_of(offset: int):
        for end, number in spans:
            if offset < end:
                return number
        return spans[-1][1] if spans else None

    return page_of


def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
    pages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []

    if split_by_character:
        sections = content.split(split_by_character)
        # Keep the separator on the section it introduces: splitting on "\nArticle " must not
        # drop the article heading. Whitespace separators vanish with the strip() below anyway.
        sections = [sec if i == 0 else split_by_character + sec for i, sec in enumerate(sections)]
        page_of = _page_lookup(tokenizer, pages)
        new_chunks = []  # (token count, text, token offset of the chunk in the document)
        offset = 0
        for section in sections:
            _tokens = tokenizer.encode(section)
            if len(_tokens) > chunk_token_size:
                if split_by_character_only:
                    logger.warning(
                        "Chunk split_by_character exceeds token limit: len=%d limit=%d",
                        len(_tokens),
                        chunk_token_size,
                    )
                    raise ChunkTokenLimitExceededError(
                        chunk_tokens=len(_tokens),
                        chunk_token_limit=chunk_token_size,
                        chunk_preview=section[:120],
                    )
                for sub in range(0, len(_tokens), chunk_token_size - chunk_overlap_token_size):
                    new_chunks.append((
                        min(chunk_token_size, len(_tokens) - sub),
                        tokenizer.decode(_tokens[sub : sub + chunk_token_size]),
                        offset + sub,
                    ))
            else:
                new_chunks.append((len(_tokens), section, offset))
            offset += len(_tokens)

        index = 0
        for _len, chunk, chunk_offset in new_chunks:
            if not chunk.strip():
                continue
            chunk_data = {"tokens": _len, "content": chunk.strip(), "chunk_order_index": index}
            page_start = page_of(chunk_offset)
            if page_start is not None:
                chunk_data["page_start"] = page_start
            results.append(chunk_data)
            index += 1

    else:
        # Token-based chunking (Standard)
        # 1. Map tokens to pages if pages provided
        token_to_page_map = []
        if pages:
            current_token_idx = 0
            for page in pages:
                # Re-encode to ensure consistent tokenization with the Full Text (assuming Full Text is concat of Pages)
                # Note: Concatenating text and encoding might produce different tokens than encoding separate texts and concatenating tokens 
                # (due to subword merging at boundaries).
                # Ideally, content should be reconstructed from pages. 
                # Assuming 'content' passed in is matching the 'pages' content.
                p_content = page.get("content", "")
                p_tokens = tokenizer.encode(p_content) # This might slightly differ from chunks of full encode
                p_len = len(p_tokens)
                
                # To be safer, we should really use character offsets if possible, but we are working with tokens here.
                # Let's hope the token count is close enough or use the token count from the FULL encoding distributed over pages proportional to length? No, that's guessing.
                
                # Let's trust that sum(len(encode(p))) ~ len(encode(sum(p))).
                # It is usually true except for the merge at the boundary.
                token_to_page_map.append((current_token_idx, current_token_idx + p_len, page.get("page_number")))
                current_token_idx += p_len
            
        for index, start in enumerate(
            range(0, len(tokens), chunk_token_size - chunk_overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + chunk_token_size])
            
            chunk_data = {
                "tokens": min(chunk_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
            
            page_start = None
            if pages and token_to_page_map:
                # Find page for 'start' token index
                # Since token counts might mismatch slightly, we scan.
                # But we can just iterate.
                for p_start, p_end, p_num in token_to_page_map:
                    if p_start <= start: # We want the page containing the start token
                         # Update candidate, but keep checking (start could be in later pages? No, sorted)
                         # p_start <= start. We want the LAST one where p_start <= start? 
                         # No. We we want the one where start < p_end.
                         if start < p_end:
                             page_start = p_num
                             break
                
            if page_start is not None:
                chunk_data["page_start"] = page_start
            
            results.append(chunk_data)

    return results

