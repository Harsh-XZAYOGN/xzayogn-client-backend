from dotenv import load_dotenv
from app.tools.base.abstract import JobSearchAPI
import os
import requests
import logging
import json
from app.schemas.models import JobData
from typing import List, Tuple, Dict, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class Web3CareerClient(JobSearchAPI):
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    def search_jobs(self, keywords: str, location: Optional[str] = None, pagesize: int = 6) -> Tuple[Optional[str], List[JobData]]:
        try:
            params = {"keywords": keywords, "page": "1", "token": self.api_key}
            if location:
                params["location"] = location

            response = requests.get(self.base_url, params=params)
            response.raise_for_status()

            jobs = [
                JobData(
                    title=job.get("title", ""),
                    company=job.get("company", ""),
                    location=job.get("location", ""),
                    url=job.get("url", ""),
                    posted_date=job.get("posted_date", ""),
                    source="web3career",
                    job_type="",
                    description=job.get("description", "")
                )
                for job in response.json().get("jobs", [])
            ]
            return None, jobs  # No error

        except requests.RequestException as e:
            logger.error(f"Web3Career API error: {e}")
            return str(e), []  # Error message and empty response

