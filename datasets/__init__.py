from .blender import BlenderDataset
from .blender_online import BlenderDatasetOnline
from .llff import LLFFDataset

dataset_dict = {'blender': BlenderDataset,
                'blender_online': BlenderDatasetOnline,
                'llff': LLFFDataset}
