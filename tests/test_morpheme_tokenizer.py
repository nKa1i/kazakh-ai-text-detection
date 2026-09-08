import unittest

class TestMorphemeTokenizer(unittest.TestCase):
    def test_morpheme_vocab_and_tokenization(self):
        from api.morpheme_tokenizer import MorphemeTokenizer
        tok = MorphemeTokenizer()
        self.assertGreater(len(tok.vocab), 50)
        self.assertIn("<PAD>", tok.vocab)
        self.assertIn("<UNK>", tok.vocab)
        self.assertIn("-лар", tok.vocab)
        self.assertIn("-ден", tok.vocab)
        self.assertIn("-сы", tok.vocab)

        sample = "Каспиден доставкасы өте тез болды"
        ids = tok.encode(sample)
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

        # Batch encoding with padding
        batch = [sample, "жақсы"]
        tensor = tok.batch_encode(batch, max_length=16)
        self.assertEqual(tensor.shape, (2, 16))
        self.assertEqual(tensor[1, -1].item(), tok.pad_token_id)

if __name__ == "__main__":
    unittest.main()
