from pathlib import Path
import tempfile
import pandas as pd

from PIL import Image
from src.data.dataset import ImageManifestDataset


def test_manifest_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create mock image
        img_dir = tmp_path / "data" / "raw" / "authentic"
        img_dir.mkdir(parents=True)
        img_file = img_dir / "test.jpg"
        Image.new("RGB", (32, 32), color="red").save(img_file)

        # Create mock manifest
        manifest_dir = tmp_path / "data" / "processed"
        manifest_dir.mkdir(parents=True)
        manifest_file = manifest_dir / "manifest.csv"

        df = pd.DataFrame([{
            "path": "data/raw/authentic/test.jpg",
            "label": 0,
            "split": "train",
            "dataset": "SID_Set",
            "source_id": "test",
            "width": 32,
            "height": 32,
            "sha256": "dummyhash"
        }])
        df.to_csv(manifest_file, index=False)

        dataset = ImageManifestDataset(manifest_path=manifest_file, split="train", root_dir=tmp_path)
        assert len(dataset) == 1
        img, label = dataset[0]
        assert label == 0
        assert img.size == (32, 32)