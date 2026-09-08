import unittest

class TestSupConLoss(unittest.TestCase):
    def test_initialization_and_graceful_import(self):
        from models.losses import SupConLoss
        criterion = SupConLoss(temperature=0.07)
        self.assertEqual(criterion.temperature, 0.07)
        self.assertTrue(callable(criterion))

    def test_fallback_forward_callable(self):
        from models.losses import SupConLoss
        criterion = SupConLoss(temperature=0.07)
        loss = criterion(None, None)
        self.assertEqual(loss.item(), 0.0)
        # Verify backward() doesn't raise exception
        loss.backward()

    def test_loss_computation_and_gradients(self):
        from models.losses import SupConLoss
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed locally; test runs on GPU environment")

        criterion = SupConLoss(temperature=0.07)
        features = torch.randn(8, 128, requires_grad=True)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])

        loss = criterion(features, labels)
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(features.grad)

    def test_identical_features_minimize_loss(self):
        from models.losses import SupConLoss
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed locally; test runs on GPU environment")

        criterion = SupConLoss(temperature=0.07)
        f1 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        f2 = torch.tensor([[-1.0, 0.0], [-1.0, 0.0]])
        features = torch.cat([f1, f2], dim=0)
        labels = torch.tensor([0, 0, 1, 1])
        loss_perfect = criterion(features, labels)

        features_random = torch.randn(4, 2)
        features_random = torch.nn.functional.normalize(features_random, p=2, dim=1)
        loss_random = criterion(features_random, labels)
        self.assertLess(loss_perfect.item(), loss_random.item())

if __name__ == "__main__":
    unittest.main()
