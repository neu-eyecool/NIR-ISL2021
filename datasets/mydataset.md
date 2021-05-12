# Dataset

## CASIA
1. The datasets are from NIR Iris Challenge Evaluation in Non-cooperative Environments: Segmentation and Localization (NIR-ISL 2021), visit https://sites.google.com/view/nir-isl2021/home for details of the competition and the dataset.
2. Before train you own models, please insure your data folader structure as follow:
    CASIA-dataset
    |--train  
        |--CASIA-Iris-Mobile-V1.0  
            |--image  
            |--SegmentationClass  
            |--iris_edge  
            |--iris_edge_mask  
            |--pupil_edge  
            |--pupil_edge_mask  
        |--CASIA-Iris-Asia  
            |--CASIA-distance  
                |--image  
                |--SegmentationClass  
                |--iris_edge  
                |--iris_edge_mask  
                |--pupil_edge  
                |--pupil_edge_mask  
            |--CASIA-Iris-Complex  
       |--CASIA-Iris-Africa  
    |--test  
       |--CASIA-Iris-Mobile-V1.0  
          |--image  
       |--CASIA-Iris-Asia  
            |--CASIA-distance  
                |--image  
            |--CASIA-Iris-Complex  
       |--CASIA-Iris-Africa  
3. Set the path (*root*) of *CASIA-dataset* folder in the last step in *NIRISL.py*