import unittest

class TestMorphoDetector(unittest.TestCase):
    def test_class_structure_and_export(self):
        from models import MorphoContrastiveDetector, MorphemeEncoder
        from models.morpho_contrastive_detector import (
            MorphoContrastiveDetector as MCD,
            MorphemeEncoder as ME
        )
        self.assertIs(MorphoContrastiveDetector, MCD)
        self.assertIs(MorphemeEncoder, ME)

    def test_initialization_defaults_and_attributes(self):
        from models.morpho_contrastive_detector import MorphoContrastiveDetector, MorphemeEncoder
        model = MorphoContrastiveDetector(
            roberta_model_name="kz-transformers/kaz-roberta-conversational",
            morpheme_vocab_size=100,
            embed_dim=768,
            proj_dim=128,
            lambda_supcon=0.5,
            temperature=0.07
        )
        self.assertEqual(model.roberta_model_name, "kz-transformers/kaz-roberta-conversational")
        self.assertEqual(model.morpheme_vocab_size, 100)
        self.assertEqual(model.embed_dim, 768)
        self.assertEqual(model.proj_dim, 128)
        self.assertEqual(model.lambda_supcon, 0.5)
        self.assertEqual(model.temperature, 0.07)
        self.assertTrue(callable(model))

        encoder = MorphemeEncoder(vocab_size=100, embed_dim=768)
        self.assertEqual(encoder.vocab_size, 100)
        self.assertEqual(encoder.embed_dim, 768)
        self.assertTrue(callable(encoder))

    def test_fallback_forward_callable(self):
        from models.morpho_contrastive_detector import MorphoContrastiveDetector, MorphemeEncoder
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False

        if has_torch:
            self.skipTest("Fallback test is for environments without torch")

        model = MorphoContrastiveDetector(
            morpheme_vocab_size=100,
            embed_dim=768,
            proj_dim=128
        )
        batch_size = 4
        input_ids = [[1] * 16] * batch_size
        attention_mask = [[1] * 16] * batch_size
        morpheme_ids = [[2] * 16] * batch_size
        labels = [0, 1, 0, 1]

        out = model(input_ids, attention_mask, morpheme_ids, labels=labels)
        self.assertIn("logits", out)
        self.assertIn("proj", out)
        self.assertIn("gate", out)
        self.assertIn("h_fused", out)
        self.assertIn("loss", out)
        self.assertIn("ce_loss", out)
        self.assertIn("supcon_loss", out)

        self.assertEqual(out["logits"].shape, (batch_size, 2))
        self.assertEqual(out["proj"].shape, (batch_size, 128))
        self.assertEqual(out["gate"].shape, (batch_size, 768))
        self.assertEqual(out["h_fused"].shape, (batch_size, 768))
        self.assertEqual(out["loss"].item(), 0.0)

        # Unlabeled forward test
        out_unlabeled = model(input_ids, attention_mask, morpheme_ids)
        self.assertNotIn("loss", out_unlabeled)
        self.assertIn("logits", out_unlabeled)

        # Fallback predict
        pred = model.predict("Керемет өнім!")
        self.assertIn("prediction", pred)
        self.assertIn("probability", pred)
        self.assertIn("gate_semantic_weight", pred)

        # Fallback MorphemeEncoder
        encoder = MorphemeEncoder(vocab_size=100, embed_dim=768)
        h_morph = encoder(morpheme_ids)
        self.assertEqual(h_morph.shape, (batch_size, 768))

    def test_morpheme_encoder_torch(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed locally; test runs on GPU environment")

        from models.morpho_contrastive_detector import MorphemeEncoder
        encoder = MorphemeEncoder(vocab_size=100, embed_dim=768, num_layers=2)
        morpheme_ids = torch.randint(0, 100, (4, 16))
        h_morph = encoder(morpheme_ids)
        self.assertEqual(h_morph.shape, (4, 768))

    def test_forward_pass_and_shapes_torch(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self.skipTest("torch not installed locally; test runs on GPU environment")

        from models.morpho_contrastive_detector import MorphoContrastiveDetector

        class DummyRoberta(nn.Module):
            def __init__(self, hidden_size=768):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_ids, attention_mask=None):
                b, s = input_ids.shape
                lhs = torch.randn(b, s, self.hidden_size, requires_grad=True)
                class DummyOutput:
                    def __init__(self, state):
                        self.last_hidden_state = state
                return DummyOutput(lhs)

        dummy_roberta = DummyRoberta(hidden_size=768)
        model = MorphoContrastiveDetector(
            roberta_model=dummy_roberta,
            morpheme_vocab_size=100,
            embed_dim=768,
            proj_dim=128,
            lambda_supcon=0.5,
            temperature=0.07
        )

        batch_size = 4
        input_ids = torch.randint(0, 1000, (batch_size, 16))
        attention_mask = torch.ones((batch_size, 16))
        morpheme_ids = torch.randint(0, 100, (batch_size, 16))
        labels = torch.tensor([0, 1, 0, 1])

        out = model(input_ids, attention_mask, morpheme_ids, labels=labels)
        self.assertIn("logits", out)
        self.assertIn("proj", out)
        self.assertIn("gate", out)
        self.assertIn("h_fused", out)
        self.assertIn("loss", out)
        self.assertIn("ce_loss", out)
        self.assertIn("supcon_loss", out)

        self.assertEqual(out["logits"].shape, (batch_size, 2))
        self.assertEqual(out["proj"].shape, (batch_size, 128))
        self.assertEqual(out["gate"].shape, (batch_size, 768))
        self.assertEqual(out["h_fused"].shape, (batch_size, 768))
        self.assertGreater(out["loss"].item(), 0.0)

        # Verify proj is L2-normalized
        proj_norms = torch.norm(out["proj"], p=2, dim=-1)
        self.assertTrue(torch.allclose(proj_norms, torch.ones(batch_size), atol=1e-5))

        # Verify gradient backprop
        out["loss"].backward()

        # Unlabeled forward test with torch
        out_unlabeled = model(input_ids, attention_mask, morpheme_ids)
        self.assertNotIn("loss", out_unlabeled)
        self.assertEqual(out_unlabeled["logits"].shape, (batch_size, 2))

if __name__ == "__main__":
    unittest.main()
