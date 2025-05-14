import faiss
import os
from pathlib import Path
import re

# Set directory and prefix
index_dir = Path("data/wiki/index")
index_prefix = "index_with_title.faiss_part_"
output_index_path = index_dir / "index_with_title.faiss"

# Sort by part number
index_paths = sorted(
    index_dir.glob(f"{index_prefix}*.index"),
    key=lambda p: int(re.search(r"part_(\d+)", p.name).group(1))
)

# Create sharded index wrapper
dim = 768  # or whatever your embedder's dimension is
index = faiss.IndexShards(dim)

# Load each shard into the wrapper
for path in index_paths:
    print(f"Adding {path.name} to sharded index")
    shard = faiss.read_index(str(path))
    index.add_shard(shard)

# Save the merged view (still full index in RAM)
faiss.write_index(index, str(output_index_path))
print(f"✅ Merged {len(index_paths)} shards into {output_index_path}")
