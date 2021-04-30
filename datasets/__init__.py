from .blender import BlenderDataset
from .llff import LLFFDataset
from .blender_new1 import BlenderDatasetNew1

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'blender_new1': BlenderDatasetNew1}