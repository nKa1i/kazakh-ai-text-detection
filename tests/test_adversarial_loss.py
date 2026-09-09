import unittest


class TestAdversarialLoss(unittest.TestCase):
    def test_invariance_loss_init_and_fallback(self):
        from models.losses import InvarianceLoss
        loss_fn = InvarianceLoss()
        self.assertTrue(callable(loss_fn))
        # Fallback when torch is not available or passing None
        res = loss_fn(None, None)
        self.assertEqual(res.item(), 0.0)
        res.backward()

    def test_invariance_loss_computation(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed locally; test runs on GPU environment")

        from models.losses import InvarianceLoss
        loss_fn = InvarianceLoss()
        z_clean = torch.randn(4, 128, requires_grad=True)
        z_adv = torch.randn(4, 128, requires_grad=True)
        loss = loss_fn(z_clean, z_adv)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(z_clean.grad)
        self.assertIsNotNone(z_adv.grad)

    def test_identical_embeddings_zero_invariance_loss(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed locally; test runs on GPU environment")

        from models.losses import InvarianceLoss
        loss_fn = InvarianceLoss()
        z = torch.randn(4, 128)
        loss = loss_fn(z, z)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_gate_logging_interface(self):
        from models.morpho_contrastive_detector import MorphoContrastiveDetector
        model = MorphoContrastiveDetector(
            morpheme_vocab_size=100,
            embed_dim=64,
            proj_dim=32,
            roberta_model=None
        )
        out = model(input_ids=[1, 2, 3], morpheme_ids=[1, 2])
        self.assertIn("gate", out)
        self.assertIn("proj", out)
        self.assertIn("logits", out)
        self.assertIn("h_fused", out)

    def test_adversarial_forward_and_gate_shift(self):
        from models.morpho_contrastive_detector import MorphoContrastiveDetector
        model = MorphoContrastiveDetector(
            morpheme_vocab_size=100,
            embed_dim=64,
            proj_dim=32,
            lambda_inv=0.5,
            roberta_model=None
        )
        out = model(
            input_ids=[1, 2, 3],
            morpheme_ids=[1, 2],
            adv_input_ids=[1, 4, 3],
            adv_morpheme_ids=[1, 5],
            labels=[0]
        )
        self.assertIn("gate", out)
        self.assertIn("gate_adv", out)
        self.assertIn("gate_diff", out)
        self.assertIn("proj", out)
        self.assertIn("proj_adv", out)
        self.assertIn("inv_loss", out)
        self.assertIn("loss", out)



if __name__ == "__main__":
    unittest.main()
