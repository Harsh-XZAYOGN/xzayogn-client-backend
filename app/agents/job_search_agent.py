import torch
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
from transformers import pipeline
from langgraph.graph import StateGraph, END, START
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from langchain.memory import ConversationBufferMemory
from app.utils.job_parser import JobQueryParser  
from app.utils.llm_helper import LLMHelper  
from app.tools.CareerJetAPI import CareerjetClient
from app.tools.Green_house import GreenhouseJobClient
from app.tools.Web3Career import Web3CareerClient
from app.tools.Jooble import JoobleClient
from app.schemas.models import JobData, AgentState

logger = logging.getLogger(__name__)

class JobSearchAgent:
    def __init__(self, pagesize: int = 6):
        """
        Initialize job search agent with LLM-powered enhancements.
        """
        self.pagesize = pagesize
        self.logger = logging.getLogger(__name__)
        self.llm = LLMHelper()  # LLM for query refinement & summarization
        self.parser = JobQueryParser()  # LangChain-based query parser
        self.memory = ConversationBufferMemory(memory_key="chat_history")  # Track user interactions
        
        # Initialize job search clients
        self.careerjet_client = CareerjetClient()
        self.greenhouse_client = GreenhouseJobClient()
        self.jooble_client = JoobleClient()
        self.web3career_client = Web3CareerClient()

    def recommend_jobs(self, user_query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Handles job search and interactive chat.
        """
        # Step 1: Handle greetings and casual conversations
        small_talk_responses = {
            "hi": "Hello! How can I assist you with your job search?",
            "hello": "Hi there! What kind of job are you looking for?",
            "how are you": "I'm an AI assistant, but I'm here to help you find jobs!"
        }
        normalized_query = user_query.lower().strip()
        if normalized_query in small_talk_responses:
            return {"message": small_talk_responses[normalized_query]}

        # Step 2: Parse user query with LangChain
        parsed_query = self.parser.parse_query(user_query)

        # Step 3: If job title is missing, ask follow-up question
        if not parsed_query.get("job_title"):
            return {"message": "What kind of job are you looking for?"}

        # Step 4: Fetch previous preferences if user has interacted before
        chat_history = self.memory.load_memory_variables({"user_id": user_id}) if user_id else {}

        # Step 5: Save user input in memory
        self.memory.save_context({"user_id": user_id}, {"query": user_query})

        # Step 6: Call job search
        results = self.search_jobs(user_query, user_profile=chat_history.get("preferences", {}))
        
        return results

    def search_jobs(self, user_query: str, user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Searches jobs based on user query & profile.
        """
        # Step 1: Parse query using LangChain
        parsed_query = self.parser.parse_query(user_query)
        job_title = parsed_query.get("job_title")
        location = parsed_query.get("location")
        max_age = parsed_query.get("max_age", None)
        num_jobs = parsed_query.get("num_jobs", self.pagesize)

        # Step 2: Personalize search if user profile exists
        if user_profile:
            location = user_profile.get("preferred_location", location)
            skills = user_profile.get("skills", parsed_query.get("skills", []))
        else:
            skills = parsed_query.get("skills", [])

        # Step 3: Call job search APIs
        all_jobs = []
        sources = [self.careerjet_client, self.greenhouse_client, self.jooble_client, self.web3career_client]

        for source in sources:
            error, jobs = source.search_jobs(job_title, location)
            if error:
                self.logger.error(f"Error fetching from {source.__class__.__name__}: {error}")
                continue
            all_jobs.extend(jobs)

        # Step 4: Filter jobs by posting date
        if max_age:
            cutoff_date = datetime.now() - timedelta(days=max_age)
            all_jobs = [job for job in all_jobs if self._parse_date(job.posted_date) >= cutoff_date]

        # Step 5: Summarize job descriptions using GPT-4
        for job in all_jobs[:num_jobs]:
            job.description = self.llm.summarize_job(job.description)

        return {"total_jobs": len(all_jobs[:num_jobs]), "jobs": all_jobs[:num_jobs]}

    def _parse_date(self, date_str: Optional[str]) -> datetime:
        """Parses job posting date or defaults to today."""
        if not date_str:
            return datetime.now()
        try:
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return datetime.now()
        except Exception:
            return datetime.now()

    def api_fetcher(self, state: AgentState) -> AgentState:
        """Fetch job data from multiple sources"""
        is_job_related = self.parser.parse_query(state.get("query", "")).get("job_title") is not None
        state["is_job_query"] = is_job_related
        state.setdefault("api_exhausted", False)
        state.setdefault("data", [])

        if not is_job_related:
            state["api_exhausted"] = True
            return state

        try:
            query = state["query"]
            search_results = self.search_jobs(query)
    
            if search_results and isinstance(search_results, dict):
                state["data"] = search_results.get("jobs", [])
                state["api_exhausted"] = len(state["data"]) == 0
                state["sources_used"] = search_results.get("sources_used", [])
                
                self.logger.info(f"Found {len(state['data'])} jobs from {len(state['sources_used'])} sources")
            else:
                state["api_exhausted"] = True
                state["data"] = []
                self.logger.warning("No results returned from search_jobs")
    
        except Exception as e:
            self.logger.error(f"API fetcher error: {str(e)}")
            state["api_exhausted"] = True
            state["errors"] = [str(e)]
    
        return state
        next_step
    )

    return agent, workflow.compile()
