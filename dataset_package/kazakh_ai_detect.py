# coding=utf-8
# Copyright 2026 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KazAI-Detect: A Multi-Domain Benchmark Dataset for AI-Generated Kazakh Text Detection."""

import csv
import os
import datasets

_DESCRIPTION = """\
KazAI-Detect is a benchmark dataset for detecting AI-generated text in the Kazakh language.
It contains authentic human texts (from KazSAnDRA consumer reviews, formal news, and Kazakh Wikipedia)
paired with synthetic machine-generated texts produced by Kazakh-adapted LLMs (such as Sherkala-7B).
"""

_HOMEPAGE = "https://github.com/dauletanekesh/kazakh-ai-text-detection"
_LICENSE = "MIT"
_CITATION = """\
@inproceedings{anekesh2026detecting,
  title={Detecting AI-Generated User Reviews in Kazakh: A Study on BERT Model Performance and False Positive Reduction},
  author={Anekesh, Daulet and Ualiyeva, Irina},
  booktitle={Proceedings of the Analysis of Images, Social Networks and Texts (AIST 2026)},
  series={Lecture Notes in Computer Science (LNCS)},
  publisher={Springer},
  year={2026}
}
"""

class KazAIDetectConfig(datasets.BuilderConfig):
    """BuilderConfig for KazAI-Detect."""
    def __init__(self, **kwargs):
        super(KazAIDetectConfig, self).__init__(version=datasets.Version("1.0.0"), **kwargs)


class KazAIDetect(datasets.GeneratorBasedBuilder):
    """KazAI-Detect: Kazakh AI-Generated Text Detection Benchmark Dataset."""

    BUILDER_CONFIGS = [
        KazAIDetectConfig(
            name="default",
            description="Full benchmark dataset including in-domain reviews and out-of-distribution test sets.",
        )
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "domain": datasets.Value("string"),
                "label": datasets.ClassLabel(num_classes=2, names=["Human", "AI"]),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        test_path = os.path.join(data_dir, "test.csv")
        ood_path = os.path.join(data_dir, "ood_test.csv")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": test_path},
            ),
            datasets.SplitGenerator(
                name="ood_test",
                gen_kwargs={"filepath": ood_path},
            ),
        ]

    def _generate_examples(self, filepath):
        if not os.path.exists(filepath):
            return

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for id_, row in enumerate(reader):
                yield id_, {
                    "text": row["text"],
                    "domain": row.get("domain", "reviews"),
                    "label": int(row["label"]),
                }
