import unittest
from kaz_mage import DomainStratifiedBatchSampler
import kaz_mage.sampler as sampler_mod

class TestDomainStratifiedSampler(unittest.TestCase):
    def test_domain_batch_stratification(self):
        # Create 300 samples: 100 reviews, 100 news, 100 wiki
        domains = ["consumer_reviews"] * 100 + ["news"] * 100 + ["wikipedia"] * 100
        labels = ([0, 1] * 50) + ([0, 1] * 50) + ([0, 1] * 50)

        sampler = DomainStratifiedBatchSampler(domains, labels, batch_size=30, seed=42)
        batches = list(sampler)
        self.assertGreater(len(batches), 0)

        # Check first batch balance
        first_batch_domains = [domains[idx] for idx in batches[0]]
        self.assertEqual(len(batches[0]), 30)
        self.assertEqual(first_batch_domains.count("consumer_reviews"), 10)
        self.assertEqual(first_batch_domains.count("news"), 10)
        self.assertEqual(first_batch_domains.count("wikipedia"), 10)

    def test_label_balance_in_batches(self):
        domains = ["consumer_reviews"] * 60 + ["news"] * 60 + ["wikipedia"] * 60
        labels = ([0, 1] * 30) + ([0, 1] * 30) + ([0, 1] * 30)
        sampler = DomainStratifiedBatchSampler(domains, labels, batch_size=30, seed=42)
        batches = list(sampler)
        first_batch_labels = [labels[idx] for idx in batches[0]]
        # Should be balanced roughly 15 label 0, 15 label 1
        self.assertAlmostEqual(first_batch_labels.count(1), 15, delta=2)

    def test_deterministic_seeding(self):
        domains = ["consumer_reviews"] * 30 + ["news"] * 30 + ["wikipedia"] * 30
        labels = [0, 1] * 45
        s1 = list(DomainStratifiedBatchSampler(domains, labels, batch_size=15, seed=42))
        s2 = list(DomainStratifiedBatchSampler(domains, labels, batch_size=15, seed=42))
        self.assertEqual(s1, s2)

    def test_len_matches_batch_count(self):
        domains = ["consumer_reviews"] * 40 + ["news"] * 40 + ["wikipedia"] * 40
        labels = ([0, 1] * 20) * 3
        # drop_last=False
        s = DomainStratifiedBatchSampler(domains, labels, batch_size=18, seed=42, drop_last=False)
        batches = list(s)
        self.assertEqual(len(s), len(batches))

        # drop_last=True
        s_drop = DomainStratifiedBatchSampler(domains, labels, batch_size=18, seed=42, drop_last=True)
        batches_drop = list(s_drop)
        self.assertEqual(len(s_drop), len(batches_drop))
        self.assertTrue(len(batches_drop) <= len(batches))

    def test_set_epoch_alters_shuffling(self):
        domains = ["consumer_reviews"] * 30 + ["news"] * 30 + ["wikipedia"] * 30
        labels = [0, 1] * 45
        s = DomainStratifiedBatchSampler(domains, labels, batch_size=15, seed=42)
        s.set_epoch(0)
        b0 = list(s)
        s.set_epoch(1)
        b1 = list(s)
        self.assertNotEqual(b0, b1)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            DomainStratifiedBatchSampler(["d1"], [0, 1], batch_size=2)
        with self.assertRaises(ValueError):
            DomainStratifiedBatchSampler(["d1", "d2"], [0, 1], batch_size=0)

if __name__ == "__main__":
    unittest.main()
