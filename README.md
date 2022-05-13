# Handwriting Recognition
Authors: Charlie Albietz, Péter Varga, Panagiotis Ritas, Jan-Henner Roberg

In this repository, you will find :

    - The Style Classification folder which contains: 
    
        -Calculate_Hinge_Features.py contains all functions for calculating the hinge features
        -Classify_Char_Style.py has the main functions for calculating the dominant style of a character
        -hinge_utils.py contains all the utilities for calculating the hinge features
        -SVM_Style.py contains the setting up and preprocessing of the SVM and calculates the dominant Style of an input image

    -The Text_Segmentation folder which contains:
    
        -characterSegmentation.py contains functions that involve splitting up and locating individual characters in the located words
        -lineSegmentation.py contains functions that help segmenting the lines of an image, this is done using the astar aglorithm
        -plotting.py contains functions for plotting/debugging in the segmentation task
        -segmentation_to_recog.py contains the CNN initialization + Transformations for its input
        -wordSegmentation.py fucntions to help split up the obtained lines into seperate words
    
    -The Character_Recog folder which contains:
    
        -data_investigation.py which contains all preprocessing functions for preparing the datasets for character recognition,
        as well as splitting into train/test, resizing and padding, etc
        -main.py which is the main function for the datapreprocessing, data augmentation, etc

    -The models folder which contains:
    
        -Character_Recognition_Train.ipynb where the code for training the character recognizer is, also contains code for cross-validation
        and other model optimizing code
        -SVM.py which contain code for training and testing a SVM for style Classification

    On the top level of the repository, there is:
        - font2image.py contains the functions for putting the styl and transcript outputs into .txt files
        - launch.py from which the whole pipeline is ran, and te the three task modules are combined.
        - two files where the trained character recognizer and Style classifer are loaded from
    
    The main file is launch.py from which the pipeline is ran to get the output.
    

To run the pipeline:

    * 1) unzip .zip file (or access clone git link)
    * 2) cd to repository
    * 3) in new virtual environment (using python3.8), do pip3 install -r requirements.txt
    the packages/dependencies should be now installed.
    
    to run the pipeline:
    
    4) run launch.py by giving the folder of the input images as parsing argument, e.g python launch.py /data/images/
    
The pipeline will now be running. as output you will get:
    -a new results/ folder which will contain the .txt output files
    -There will also be a /paths folder for each image that contains the astar .csv files for line/char segmentation

The final paper can be found in the final papers folder
