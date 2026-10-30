import unittest

from probe_int.train_step_labels import probe_prompt


class ProbePromptTests(unittest.TestCase):
    def test_returns_token_id_list_with_transformers_5_default_dict_behavior(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize, return_dict,
                                    add_generation_prompt, enable_thinking):
                self.content = messages[0]["content"]
                self.return_dict = return_dict
                return {"input_ids": [1, 2, 3]} if return_dict else [1, 2, 3]

            def encode(self, text, add_special_tokens):
                return [9, 10]

        tokenizer = Tokenizer()
        prompt = probe_prompt(tokenizer, {
            "question": "Problem",
            "reference_solution": "Answer",
            "reasoning_prefix": "Work so far",
        })

        self.assertEqual(prompt, [1, 2, 3])
        self.assertFalse(tokenizer.return_dict)
        self.assertIn("Work so far", tokenizer.content)


if __name__ == "__main__":
    unittest.main()
