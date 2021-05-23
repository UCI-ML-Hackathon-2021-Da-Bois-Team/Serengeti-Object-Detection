# Serengeti-Object-Detection
This code works on the HPC3 cluster at UCI.
The code/notebooks are run in the JupyterLab, and the datasets are also downloaded in the cluster.

We are working on the Gold Standard Serengeti Dataset.

Data Preprocessing:
  - This code contains preprocessing code that converts the information from the BoundingBox json file (labelled metadata of the images).
  - It converts the data to an acceptable format for Yolov5 to train on.
  - A config file was written to be read by Yolov5 (serengeti.yaml)

Yolov5 Model:
  What we have:
  - Yolov5 is implemented to train on a small dataset of the Serengeti Data. 

  What we will work on:
  - Its hyperparameters are evaluated using its built-in genetic algorithm. 
  - The model will then trains on its optimized hyperparameters on a larger Yolov5 Data. 



