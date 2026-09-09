import unittest
import os
import py_compile

class TestKazMageKernelBuild(unittest.TestCase):
    def test_kernel_build_and_syntax(self):
        from scripts.prepare_kaz_mage_kernel import build_kaz_mage_kernel
        kernel_path = os.path.join('kaggle_runner', 'test_mage_kernel.py')
        build_kaz_mage_kernel(output_path=kernel_path, dry_run=True)
        self.assertTrue(os.path.exists(kernel_path))
        py_compile.compile(kernel_path, doraise=True)
        if os.path.exists(kernel_path):
            os.remove(kernel_path)

if __name__ == '__main__':
    unittest.main()
