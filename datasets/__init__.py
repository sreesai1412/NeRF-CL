from .blender import BlenderDataset
from .blender_large import BlenderDatasetLarge
from .blender_large_online import BlenderDatasetLargeOnline
from .llff import LLFFDataset

dataset_dict = {'blender': BlenderDataset,
                'blender_large': BlenderDatasetLarge,
                'blender_large_online': BlenderDatasetLargeOnline,
                'llff': LLFFDataset}
