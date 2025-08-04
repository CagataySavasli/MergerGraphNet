from google import genai
from google.genai import types
import time
import json
import re


class GeminiChat():
    def __init__(self, api_key: str, model_version: str, system_prompt: str, thinking_budget: int):
        self.api_key = api_key
        self.model_version = model_version
        self.system_prompt = system_prompt
        self.thinking_budget = thinking_budget

        self.client = genai.Client(api_key=api_key)



    def send_message(self, prompt):
        response = self.client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
                system_instruction=self.system_prompt,
            ),
        )

        return response

    def check_response(self, prompt):
        for time_sleep in [3, 5, 10]:
            try:
                response = self.send_message(prompt)
                return response
            except Exception as e:
                time.sleep(time_sleep)
                continue

        return None

    def get_text(self, response):
        try:
            answer_text = response.text
        except :
            answer_text = None
        return answer_text

    def parse_markdown_json(self, md_str: str) -> dict:
        """
        Extracts JSON from a markdown code fence and returns it as a Python dict.

        Args:
            md_str (str): A string containing a JSON code block, e.g.:
                '```json\n{ ... }\n```'

        Returns:
            dict: The parsed JSON object.
        """
        # Remove the opening and closing ```json fences
        # This regex strips everything before the first '{' and after the last '}'
        try:
            json_text = re.search(r'\{.*\}', md_str, flags=re.DOTALL).group(0)
            result = json.loads(json_text)
        except:
            result = None
        return result

    def predict(self, prompt):
        response = self.check_response(prompt)
        if response:
            text = self.get_text(response)
        else:
            return "ERROR"
        if text:
            answer_dict = self.parse_markdown_json(text)
        else:
            return "ERROR"
        return answer_dict