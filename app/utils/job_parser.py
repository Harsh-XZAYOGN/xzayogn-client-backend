import logging
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
from app.config.settings import settings

logger = logging.getLogger(__name__)

class JobQueryParser:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", api_key=settings.OPENAI_API_KEY, temperature=0)

    def parse_query(self, user_query: str) -> dict:
        """
        Uses LangChain structured parsing to extract job details.
        """
        response_schemas = [
            ResponseSchema(name="job_title", description="Job title being searched for"),
            ResponseSchema(name="location", description="Location for the job"),
            ResponseSchema(name="experience", description="Experience required (e.g., '3+ years')"),
            ResponseSchema(name="skills", description="List of required skills"),
            ResponseSchema(name="max_age", description="Max age of job posting in days"),
            ResponseSchema(name="num_jobs", description="Number of job listings requested")
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract structured job details from user queries."),
            ("user", "{query}\n\nFormat your response as JSON:\n" + parser.get_format_instructions())
        ])
        
        chain = prompt | self.llm | parser
        try:
            structured_output = chain.invoke({"query": user_query})
            logger.info(f"Parsed Query: {structured_output}")
            return structured_output
        except Exception as e:
            logger.error(f"Job Query Parsing Error: {e}")
            return {"job_title": None, "location": None, "experience": None, "skills": [], "max_age": None, "num_jobs": None}
