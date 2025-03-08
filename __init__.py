import os
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import folder_paths
import folder_paths

# Then import your other modules
from .IF_HyDatasetMkr import IF_HyDatasetMkr
                       

NODE_CLASS_MAPPINGS = {
    "IF_DatasetMkr": IF_HyDatasetMkr,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_DatasetMkr": "IF VideoDatasetMkr 📚",
}




WEB_DIRECTORY = "./web"
__all__ = [
    "NODE_CLASS_MAPPINGS", 
    "NODE_DISPLAY_NAME_MAPPINGS", 
    "WEB_DIRECTORY", 
    ]
