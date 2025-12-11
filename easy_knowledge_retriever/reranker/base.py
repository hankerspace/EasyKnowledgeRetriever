from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseRerankerService(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents based on a query.
        
        Args:
            query: The search query
            documents: List of document strings to rerank
            top_n: Optional number of top results to return
            
        Returns:
            List of dictionaries containing relevance scores and indices.
            Example: [{"index": 0, "relevance_score": 0.95}, ...]
        """
        pass
