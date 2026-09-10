"""Entity-variant merging and PDF text cleanup. Run: python test_ingest_quality.py"""
from easy_knowledge_retriever.operations.graph_ops import entity_merge_key
from easy_knowledge_retriever.operations.mineru_parser import clean_extracted_text

assert entity_merge_key("High-Risk AI Systems") == entity_merge_key("High-risk  AI system")
assert entity_merge_key("Systèmes d'IA à haut risque") == entity_merge_key("système d'IA à haut risque")
assert entity_merge_key("Commission") != entity_merge_key("Comité")
assert entity_merge_key("IA") == "ia"  # short words keep their last letter
assert clean_extracted_text("l'article $^ { 16 ; }$ ▌autres") == "l'article 16; autres"
assert clean_extracted_text("10<sup>25</sup> FLOP") == "10<sup>25</sup> FLOP"
print("ok")

# Article-aware chunking keeps headings and the page each chunk starts on.
from easy_knowledge_retriever.operations.chunking import chunking_by_token_size
from easy_knowledge_retriever.utils.tokenizer import TiktokenTokenizer

pages = [{"content": "Considérant liminaire.\n", "page_number": 1},
         {"content": "Article 1\nObjet du règlement.\nArticle 2\nChamp d'application.\n", "page_number": 2}]
text = "".join(p["content"] for p in pages)
chunks = chunking_by_token_size(TiktokenTokenizer(), text, "\nArticle ", False, 10, 600, pages=pages)
assert [c["content"].split("\n")[0] for c in chunks] == ["Considérant liminaire.", "Article 1", "Article 2"], chunks
assert [c.get("page_start") for c in chunks] == [1, 2, 2], chunks
long_chunks = chunking_by_token_size(TiktokenTokenizer(), "mot " * 1500, None, False, 50, 600, pages=[{"content": "mot " * 1500, "page_number": 7}])
assert len(long_chunks) == 3 and long_chunks[0]["page_start"] == 7
print("chunking ok")
