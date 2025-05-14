import faiss
import os
from pathlib import Path

# Set the directory containing the partial indexes
index_dir = Path("data/wiki/index")
index_prefix = "index_with_title.faiss_part_"
output_index_path = index_dir / "index_with_title.faiss"

# Collect all partial index paths and sort by part number
index_paths = sorted(index_dir.glob(f"{index_prefix}*.index"), key=lambda p: int(p.stem.split("_")[-2]))

# Read the first index as destination
index = faiss.read_index(str(index_paths[0]))

# Read the rest and merge
to_merge = [faiss.read_index(str(p)) for p in index_paths[1:]]
faiss.merge_into(index, to_merge, shift_ids=False)

# Save the merged index
faiss.write_index(index, str(output_index_path))
print(f"✅ Merged {len(index_paths)} shards into {output_index_path}")
