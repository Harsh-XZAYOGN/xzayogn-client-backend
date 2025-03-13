from app.tools.base.abstract import JobSearchAPI
import os
import requests
import logging
import json
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List, Tuple, Optional
from app.schemas.models import JobData

logger = logging.getLogger(__name__)

class JoobleClient(JobSearchAPI):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = f"https://jooble.org/api/{self.api_key}"

    def search_jobs(self, keywords: str, location: Optional[str] = None, pagesize: int = 6) -> Tuple[Optional[str], List[JobData]]:
        try:
            payload = {
                "keywords": keywords,
                "location": location or "",
                "pagesize": pagesize
            }
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()

            jobs = [
                JobData(
                    title=job.get("title", ""),
                    company=job.get("company", ""),
                    location=job.get("location", ""),
                    url=job.get("url", ""),
                    posted_date=job.get("posted_date", ""),
                    source="jooble",
                    job_type="",
                    description=""
                )
                for job in response.json().get("jobs", [])
            ]
            return None, jobs  # No error

        except requests.RequestException as e:
            logger.error(f"Jooble API error: {e}")
            return str(e), []  # Error message and empty response
