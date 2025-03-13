from pydantic import BaseModel, Field
from app.tools.base.abstract import JobSearchAPI
from careerjet_api_client import CareerjetAPIClient
import logging
from typing import List, Tuple, Dict, Optional
import json
from app.schemas.models import JobData
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)

class CareerjetClient(JobSearchAPI):
    def __init__(self, affid='cd918e610ecfbcd6cc50f9527541794c'):
        self.cj_client = CareerjetAPIClient("en_US")
        self.affid = affid

    def search_jobs(self, keywords, location=None, pagesize=6) -> Tuple[Optional[str], List[JobData]]:
        try:
            search_params = {
                'keywords': keywords,
                'location': location or '',
                'affid': self.affid,
                'user_ip': '11.22.33.44',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'url': 'https://www.example.com/jobs',
                'pagesize': pagesize
            }

            logger.info(f"Searching Careerjet with: {json.dumps(search_params, indent=2)}")
            response = self.cj_client.search(search_params)

            jobs = [
                JobData(
                    title=job.get("title", ""),
                    company=job.get("company", ""),
                    location=job.get("locations", ""),
                    url=job.get("url", ""),
                    posted_date=job.get("date", ""),
                    source="careerjet",
                    job_type="",
                    description=job.get("description", "")
                )
                for job in response.get("jobs", [])
            ]
            return None, jobs  # No error

        except Exception as e:
            logger.error(f"Careerjet API error: {e}")
            return str(e), []  # Error message and empty response
