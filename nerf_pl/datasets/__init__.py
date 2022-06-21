from .blender import BlenderDataset
from .llff import LLFFDataset
from .llff_orig import LLFFDatasetOrig

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'llff_orig': LLFFDatasetOrig}