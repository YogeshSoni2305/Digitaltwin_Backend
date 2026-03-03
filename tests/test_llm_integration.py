import json
import os
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app, STATE

class TestLLMIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Mock employees and projects for the test
        from core.models import Employee, ExecutionProject, ExecutionTask
        STATE["employees"] = [
            Employee(id="E1", name="Test User", role="Engineer", department="IT", salary=100000, 
                     skills={"backend": 0.8}, capacity_hours_per_week=40, productivity_multiplier=1.0,
                     reports_to=None),
            Employee(id="E2", name="Backup User", role="Engineer", department="IT", salary=90000, 
                     skills={"backend": 0.7}, capacity_hours_per_week=40, productivity_multiplier=1.0,
                     reports_to=None)
        ]
        # In the new main.py, STATE["projects"] is initialized in load_system_state.
        # We'll overwrite it for the test.
        STATE["projects"] = [
            ExecutionProject(id="P1", name="Test Project", base_revenue=100000, decay_rate=0.1,
                             tasks=[ExecutionTask(id="T1", name="Task 1", required_skill="backend", 
                                                 required_level=0.5, estimated_hours=10, dependencies=[])])
        ]
        os.environ["GROQ_API_KEY"] = "test_key"

    def test_explain_endpoint_logic(self):
        """Tests that the /decision/explain endpoint orchestrates logic correctly."""
        
        payload = {
            "employee_id": "E1",
            "seed": 42
        }

        # Mock Groq client and persistence in the NEW location
        with patch("core.llm.explainer.Groq") as mock_groq, \
             patch("core.persistence.storage.safe_append_json_record") as mock_append:
            
            # Mock the LLM output
            mock_client_instance = MagicMock()
            mock_groq.return_value = mock_client_instance
            mock_client_instance.chat.completions.create.return_value.choices[0].message.content = "Mocked LLM Explanation"

            response = self.client.post("/decision/explain", json=payload)
            
            if response.status_code != 200:
                print(f"DEBUG: Response error: {response.text}")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertIn("best_strategy", data)
            self.assertIn("executive_summary", data) # New field name
            self.assertIn("rankings", data)
            self.assertIn("winning_data_snapshot", data) # New field name
            
            # Verify Mocked LLM output is in response
            self.assertEqual(data["executive_summary"], "Mocked LLM Explanation")
            
            # Verify persistence was called
            self.assertTrue(mock_append.called)

    @patch("core.llm.explainer.Groq")
    def test_llm_persistence_path(self, mock_groq):
        """Verifies that the explanation is actually written to the file (using real storage)."""
        from core.llm.explainer import interpret_simulation_outcome
        
        # Setup mock
        mock_client_instance = MagicMock()
        mock_groq.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value.choices[0].message.content = "Persistence Test"
        
        # Clean up existing test file if any
        from core.persistence.storage import FILE_PATH_LLM_EXPLANATIONS
        if os.path.exists(FILE_PATH_LLM_EXPLANATIONS):
            os.remove(FILE_PATH_LLM_EXPLANATIONS)
            
        # Run explainer
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
            interpret_simulation_outcome({"test": "data"}, "strategy_a", 123)
            
        # Verify file creation and content
        self.assertTrue(os.path.exists(FILE_PATH_LLM_EXPLANATIONS))
        with open(FILE_PATH_LLM_EXPLANATIONS, "r") as f:
            data = json.load(f)
            self.assertIsInstance(data, list)
            self.assertEqual(data[0]["executive_summary"], "Persistence Test") # New field name
            self.assertEqual(data[0]["strategy"], "strategy_a")

if __name__ == "__main__":
    unittest.main()
