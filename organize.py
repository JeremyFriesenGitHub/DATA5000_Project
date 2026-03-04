#script to organize into dataset folder

import shutil
from pathlib import Path

data_dir = Path("output")  # run from landcover.ai.v1 directory
output_dir = Path("dataset")

for split in ["train", "val", "test"]:
    (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)
    
    names = [line.strip() for line in open(f"{split}.txt") if line.strip()]
    found = 0
    
    for name in names:
        img = data_dir / f"{name}.jpg"
        mask = data_dir / f"{name}_m.png"
        
        if img.exists() and mask.exists():
            shutil.copy(img, output_dir / split / "images" / img.name)
            shutil.copy(mask, output_dir / split / "masks" / mask.name)
            found += 1
    
    print(f"{split}: {found}/{len(names)} tiles")

print(f"\nDone! Dataset organized in {output_dir}")