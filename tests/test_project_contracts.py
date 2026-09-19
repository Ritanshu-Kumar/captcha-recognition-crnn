import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectContractTests(unittest.TestCase):
    def test_required_files_exist(self):
        required = [
            "README.md",
            "requirements.txt",
            "src/__init__.py",
            "src/dataset.py",
            "src/model.py",
            "src/train.py",
            "src/evaluate.py",
            "results/loss_curves.png",
            "results/confusion_matrix.png",
        ]

        for relative_path in required:
            self.assertTrue((ROOT / relative_path).exists(), relative_path)

    def test_python_sources_parse(self):
        source_files = [
            ROOT / "src" / "dataset.py",
            ROOT / "src" / "model.py",
            ROOT / "src" / "train.py",
            ROOT / "src" / "evaluate.py",
        ]

        for path in source_files:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))

    def test_dataset_contract(self):
        source = (ROOT / "src" / "dataset.py").read_text(encoding="utf-8")

        for expected in [
            "IMG_HEIGHT = 64",
            "IMG_WIDTH = 200",
            "BATCH_SIZE = 16",
            "TEST_SIZE = 0.20",
            "RANDOM_STATE = 0",
            "train_test_split(",
        ]:
            self.assertIn(expected, source)

    def test_model_contract(self):
        source = (ROOT / "src" / "model.py").read_text(encoding="utf-8")

        for expected in [
            "resnet18",
            "nn.GRU",
            "bidirectional=True",
            "CTC",
        ]:
            self.assertIn(expected, source)

    def test_training_contract(self):
        source = (ROOT / "src" / "train.py").read_text(encoding="utf-8")

        for expected in [
            "NUM_EPOCHS = 30",
            "LEARNING_RATE = 0.001",
            "WEIGHT_DECAY = 1e-3",
            "nn.CTCLoss",
            "clip_grad_norm_",
        ]:
            self.assertIn(expected, source)

    def test_evaluation_contract(self):
        source = (ROOT / "src" / "evaluate.py").read_text(encoding="utf-8")

        for expected in [
            "accuracy_score",
            "precision_score",
            "recall_score",
            "f1_score",
            "confusion_matrix",
            "predictions.csv",
            "metrics.txt",
        ]:
            self.assertIn(expected, source)

    def test_readme_does_not_reference_missing_architecture_image(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("docs/architecture.png", readme)


if __name__ == "__main__":
    unittest.main()
