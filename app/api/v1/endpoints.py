from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.agents.job_search_agent import JobSearchAgent
from app.database import get_user_profile  # Function to fetch user profile from DB

router = APIRouter()
job_agent = JobSearchAgent()

@router.post("/recommend-jobs/")
async def recommend_jobs(user_query: str, user_id: Optional[str] = None):
    """
    Recommend jobs based on user query.
    - If `user_id` is provided, fetch user profile for personalized recommendations.
    - Otherwise, return general job recommendations.
    """
    user_profile = get_user_profile(user_id) if user_id else None
    results = job_agent.recommend_jobs(user_query, user_profile)
    
    if not results["jobs"]:
        raise HTTPException(status_code=404, detail="No jobs found matching the criteria.")

    return results
