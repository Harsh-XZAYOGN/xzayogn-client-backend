import unittest
from unittest.mock import patch, MagicMock
from backend.app.schemas.models import JobData, SearchResponse
from backend.app.tools.CareerJetAPI import CareerjetClient
from backend.app.tools.Jooble import JoobleClient
from backend.app.tools.Web3Career import Web3CareerClient
from backend.app.agents.job_search_agent import JobSearchAgent
import requests
import time

class TestJobData(unittest.TestCase):
    def test_job_data_serialization(self):
        job = JobData(
            title="Software Engineer",
            job_type="Full-time",
            description="Develop applications",
            posted_date="2024-03-12",
            company="TechCorp",
            location="New York, USA",
            url="https://example.com/job",
            source="careerjet"
        )
        self.assertEqual(job.to_dict()["title"], "Software Engineer")
        self.assertEqual(job.to_dict()["source"], "careerjet")

    def test_job_data_missing_fields(self):
        job = JobData()
        self.assertEqual(job.title, "")
        self.assertEqual(job.source, "careerjet")

class TestCareerjetClient(unittest.TestCase):
    @patch("backend.app.tools.CareerJetAPI.CareerjetClient.search_jobs")
    def test_search_jobs(self, mock_search_jobs):
        mock_search_jobs.return_value = (None, [JobData(title="Dev", company="X Corp")])
        client = CareerjetClient()
        err, jobs = client.search_jobs("developer")
        self.assertIsNone(err)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].title, "Dev")
    
    @patch("backend.app.tools.CareerJetAPI.CareerjetClient.search_jobs")
    def test_careerjet_api_failure(self, mock_search_jobs):
        mock_search_jobs.return_value = ("API Error", [])
        client = CareerjetClient()
        err, jobs = client.search_jobs("developer")
        self.assertIsNotNone(err)
        self.assertEqual(jobs, [])
    
    @patch("requests.post")
    def test_careerjet_missing_jobs_key(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {}  # Missing 'jobs' key
        client = CareerjetClient()
        err, jobs = client.search_jobs("developer")
        self.assertIsNone(err)
        self.assertEqual(jobs, [])

class TestJoobleClient(unittest.TestCase):
    @patch("requests.post")
    def test_search_jobs(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"jobs": [{"title": "Engineer", "company": "ABC"}]} 
        client = JoobleClient(api_key="test")
        err, jobs = client.search_jobs("engineer")
        self.assertIsNone(err)
        self.assertEqual(jobs[0].title, "Engineer")
    
    @patch("requests.post")
    def test_jooble_partial_response(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"jobs": [{"title": "", "company": "ABC"}]}  # Missing title
        client = JoobleClient(api_key="test")
        err, jobs = client.search_jobs("engineer")
        self.assertIsNone(err)
        self.assertEqual(jobs[0].title, "")
    
    @patch("requests.post")
    def test_jooble_timeout(self, mock_post):
        mock_post.side_effect = requests.Timeout
        client = JoobleClient(api_key="test")
        err, jobs = client.search_jobs("engineer")
        self.assertIsNotNone(err)
        self.assertEqual(jobs, [])

class TestWeb3CareerClient(unittest.TestCase):
    @patch("requests.get")
    def test_search_jobs(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"jobs": [{"title": "Blockchain Dev", "company": "CryptoCorp"}]}
        client = Web3CareerClient(api_key="test", base_url="http://web3career.com")
        err, jobs = client.search_jobs("blockchain")
        self.assertIsNone(err)
        self.assertEqual(jobs[0].title, "Blockchain Dev")
    
    @patch("requests.get")
    def test_web3career_invalid_response(self, mock_get):
        mock_get.return_value.status_code = 500
        client = Web3CareerClient(api_key="test", base_url="http://web3career.com")
        err, jobs = client.search_jobs("blockchain")
        self.assertIsNotNone(err)
        self.assertEqual(jobs, [])

class TestJobSearchAgent(unittest.TestCase):
    @patch("backend.app.agents.job_search_agent.JobSearchAgent._search_careerjet")
    def test_search_jobs(self, mock_careerjet):
        mock_careerjet.return_value = [JobData(title="AI Engineer", company="AI Labs")]
        agent = JobSearchAgent()
        results = agent.search_jobs("AI")
        self.assertEqual(results["total_jobs"], 1)
        self.assertEqual(results["jobs"][0].title, "AI Engineer")
    
    @patch("backend.app.agents.job_search_agent.JobSearchAgent.search_jobs")
    def test_empty_job_query(self, mock_search_jobs):
        mock_search_jobs.return_value = {"total_jobs": 0, "jobs": []}
        agent = JobSearchAgent()
        results = agent.search_jobs("")
        self.assertEqual(results["total_jobs"], 0)
        self.assertEqual(results["jobs"], [])
    
    @patch("backend.app.agents.job_search_agent.JobSearchAgent.search_jobs")
    def test_agent_no_results(self, mock_search_jobs):
        mock_search_jobs.return_value = {"total_jobs": 0, "jobs": []}
        agent = JobSearchAgent()
        results = agent.search_jobs("unknown job")
        self.assertEqual(results["total_jobs"], 0)
        self.assertEqual(results["jobs"], [])

if __name__ == "__main__":
    unittest.main()
