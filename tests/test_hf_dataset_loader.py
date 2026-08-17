import unittest
import os
import pandas as pd

class TestHFDatasetPackage(unittest.TestCase):

    def setUp(self):
        self.pkg_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset_package')
        self.data_dir = os.path.join(self.pkg_dir, 'data')

    def test_dataset_files_exist(self):
        self.assertTrue(os.path.exists(os.path.join(self.pkg_dir, 'kazakh_ai_detect.py')))
        self.assertTrue(os.path.exists(os.path.join(self.pkg_dir, 'README.md')))
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, 'test.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, 'ood_test.csv')))

    def test_data_schema(self):
        df_ood = pd.read_csv(os.path.join(self.data_dir, 'ood_test.csv'))
        expected_cols = {'text', 'domain', 'label'}
        self.assertTrue(expected_cols.issubset(set(df_ood.columns)))
        self.assertGreater(len(df_ood), 1000)
        self.assertTrue(set(df_ood['label'].unique()).issubset({0, 1}))

if __name__ == '__main__':
    unittest.main()
