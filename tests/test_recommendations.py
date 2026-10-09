import unittest

from src.recommendations import build_resume_recommendations


class RecommendationTests(unittest.TestCase):
    def test_generates_basic_recommendations(self):
        extracted = {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "",
            "location": "",
            "skills": ["python", "sql"],
            "experience_entries": [],
            "degrees": [],
            "job_titles": [],
        }

        result = build_resume_recommendations(extracted, "HEALTHCARE")
        self.assertIn("score", result)
        self.assertIn("ats_score", result)
        self.assertIn("missing_sections", result)
        self.assertIn("strengths", result)
        self.assertTrue(result["recommendations"])
        self.assertGreater(len(result["recommendations"]), 0)


if __name__ == "__main__":
    unittest.main()
