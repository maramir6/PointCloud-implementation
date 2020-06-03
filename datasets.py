import json
import os
import csv
import torch.utils.data as data
import torch
import pandas as pd
import numpy as np
from PIL import Image
from laspy.file import File

class LidarDataset(data.Dataset):
    
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 3
    
    def __init__(self, dataset_folder, number_of_points=2500):
        self.dataset_folder = dataset_folder
        self.number_of_points = number_of_points
        
        category_file = os.path.join(self.dataset_folder, 'CONSOLIDADO.csv')
        self.folders_to_classes_mapping = {}
        self.segmentation_classes_offset = {}
        
        df = pd.read_csv(category_file)
        #fid, prun = df['INDEX'].values.tolist(), df['POD'].values.tolist()
        prun = df['POD'].values.tolist()
        fid = [*range(len(prun))]

        self.files = df['FILE'].values.tolist()
        self.folders_to_classes_mapping = dict(zip(fid, prun))
        
    def __getitem__(self, index):
        point_file = self.files[index]
        point_cloud_class = self.folders_to_classes_mapping[index]
        return self.prepare_data(point_file, self.number_of_points, point_cloud_class=point_cloud_class)

    def __len__(self):
        return len(self.files)

    @staticmethod
    def prepare_data(point_file, number_of_points=None, point_cloud_class=None, segmentation_label_file=None, segmentation_classes_offset=None):
        
        infile = File(point_file, mode="r")
        x, y, z = infile.x, infile.y, infile.z
        
        min_x, min_y, min_z = np.min(x), np.min(y), np.min(z)
        
        x = np.array((1/0.01)*(x - min_x), dtype=np.int)
        y = np.array((1/0.01)*(y - min_y), dtype=np.int)
        z = np.array((1/0.01)*(z - min_z), dtype=np.int)
        
        point_cloud = np.stack((x, y, z), axis=1).astype(np.float32)
        
        if number_of_points:
            sampling_indices = np.random.choice(point_cloud.shape[0], number_of_points)
            point_cloud = point_cloud[sampling_indices, :]
        point_cloud = torch.from_numpy(point_cloud)
        if segmentation_label_file:
            segmentation_classes = np.loadtxt(segmentation_label_file).astype(np.int64)
            if number_of_points:
                segmentation_classes = segmentation_classes[sampling_indices]
            segmentation_classes = segmentation_classes + segmentation_classes_offset -1
            segmentation_classes = torch.from_numpy(segmentation_classes)
            return point_cloud, segmentation_classes
        elif point_cloud_class is not None:
            point_cloud_class = torch.tensor(point_cloud_class)
            return point_cloud, point_cloud_class
        else:
            return point_cloud