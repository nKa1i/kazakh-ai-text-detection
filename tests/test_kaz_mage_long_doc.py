import unittest

class TestKazMageLongDoc(unittest.TestCase):
    def test_morpheme_tokenizer_long_text_batch_encode(self):
        from api.morpheme_tokenizer import MorphemeTokenizer
        tok = MorphemeTokenizer()
        long_text = "Қазақстан Республикасының мемлекеттік тілі – қазақ тілі болып табылады. " * 25
        encoded = tok.encode(long_text)
        self.assertGreater(len(encoded), 100)

        tensor_256 = tok.batch_encode([long_text], max_length=256)
        self.assertEqual(tensor_256.shape, (1, 256))

        tensor_512 = tok.batch_encode([long_text], max_length=512)
        self.assertEqual(tensor_512.shape, (1, 512))

    def test_morpheme_encoder_pos_embedding_capacity(self):
        from models.morpho_contrastive_detector import MorphemeEncoder, HAS_TORCH
        encoder = MorphemeEncoder(vocab_size=250, embed_dim=768)
        if HAS_TORCH:
            import torch
            self.assertGreaterEqual(encoder.pos_embedding.num_embeddings, 512)
            dummy_ids = torch.randint(0, 200, (2, 300))
            out = encoder(dummy_ids)
            self.assertEqual(out.shape, (2, 768))

if __name__ == "__main__":
    unittest.main()
