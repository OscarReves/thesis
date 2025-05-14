import faiss
import os
from pathlib import Path
import re

# Set directory and prefix
index_dir = Path("data/wiki/index")
index_prefix = "index_with_title.faiss_part_"
output_index_path = index_dir / "index_with_title.faiss"

# Match files and extract part numbers using regex
index_paths = sorted(
    index_dir.glob(f"{index_prefix}*.index"),
    key=lambda p: int(re.search(r"part_(\d+)", p.name).group(1))
)

# Load the first index
index = faiss.read_index(str(index_paths[0]))

# Merge the rest
for path in index_paths[1:]:
    print(f"Merging {path.name}")
    shard = faiss.read_index(str(path))
    faiss.merge_into(index, shard, shift_ids=False)

# Save merged index
faiss.write_index(index, str(output_index_path))
print(f"✅ Merged {len(index_paths)} shards into {output_index_path}")
