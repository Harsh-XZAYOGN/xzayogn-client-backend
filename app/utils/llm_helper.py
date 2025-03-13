import openai
import logging
from app.config.settings import settings

logger = logging.getLogger(__name__)

class LLMHelper:
    """
    Uses GPT-4 to refine job search queries and summarize job descriptions.
    """

    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        openai.api_key = self.api_key  # Set API Key for OpenAI

    def refine_query(self, user_query: str) -> str:
        """
        Uses GPT-4 to refine and improve job search queries.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Refine job search queries for better accuracy."},
                    {"role": "user", "content": f"Refine this query: {user_query}"}
                ],
                temperature=0.7
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM Error in refining query: {e}")
            return user_query  # Return original query if error occurs

    def summarize_job(self, job_description: str) -> str:
        """
        Uses GPT-4 to generate a short summary of a job description.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Summarize job descriptions in 2-3 sentences."},
                    {"role": "user", "content": f"Summarize this job description: {job_description}"}
                ],
                temperature=0.5
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM Error in summarizing job: {e}")
            return job_description[:200]  # Return first 200 characters if error occurs
