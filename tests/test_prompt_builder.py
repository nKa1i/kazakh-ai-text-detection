import unittest
from scripts.prompt_builder import build_generation_prompt

class TestPromptBuilder(unittest.TestCase):
    def test_build_generation_prompt_short(self):
        seed = {
            "text": "Өте керемет тауар, тез жеткізді!",
            "char_length": 30,
            "length_bracket": "short",
            "domain": "consumer_reviews"
        }
        prompt = build_generation_prompt(seed)
        self.assertIn("Қазақ тілінде", prompt)
        self.assertIn("қысқа", prompt)
        self.assertIsInstance(prompt, str)

    def test_build_generation_prompt_medium(self):
        seed = {
            "text": "Сапасы жақсы, бірақ бағасы сәл қымбаттау екен.",
            "char_length": 70,
            "length_bracket": "medium",
            "domain": "consumer_reviews"
        }
        prompt = build_generation_prompt(seed)
        self.assertIn("орташа", prompt.lower())

    def test_build_generation_prompt_long(self):
        seed = {
            "text": "Бұл тауарды бір ай бұрын алған болатынмын...",
            "char_length": 110,
            "length_bracket": "long",
            "domain": "consumer_reviews"
        }
        prompt = build_generation_prompt(seed)
        self.assertIn("толық", prompt.lower())

if __name__ == "__main__":
    unittest.main()
