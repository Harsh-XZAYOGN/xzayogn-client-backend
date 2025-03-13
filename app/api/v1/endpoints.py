from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.agents.job_search_agent import JobSearchAgent
from app.database import get_user_profile  # Function to fetch user profile from DB

router = APIRouter()
job_agent = JobSearchAgent()

@router.post("/chat/")
async def chat(user_query: str, user_id: Optional[str] = None):
    """
    Handles interactive job search and general conversation.
    - If user asks about jobs, recommend jobs.
    - If user greets the bot, respond with a friendly message.
    - If user input is unclear, ask follow-up questions.
    """
    response = job_agent.recommend_jobs(user_query, user_id)
    return response
