from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
from app.schemas.models import JobData

class JobSearchAPI(ABC):
    """
    Abstract base class for job search APIs.
    """

    @abstractmethod
    def search_jobs(
        self, keywords: str, location: Optional[str] = None, pagesize: int = 6
    ) -> Tuple[Optional[str], List[JobData]]:
        """
        Search for jobs based on keywords and location.

        Args:
            keywords (str): Job search query.
            location (Optional[str]): Location filter.
            pagesize (int): Number of results per page.

        Returns:
            Tuple[Optional[str], List[JobData]]: 
                - error (Optional[str]): Error message if any, otherwise None.
                - response (List[JobData]): List of job listings (empty if error occurs).
        """
        pass
