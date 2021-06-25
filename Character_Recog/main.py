from data_investigation import *

####Data investigation before crop####
avg_x,avg_y = data_size_stats("HandwritingRecog/data/Char_Recog/hhd_dataset/hdd_dataset/TRAIN/*/*.png")

#####Get stats for cropped images####
#max_x,max_y, img_size_all_x, img_size_all_y = CroppedCharAnalysis("HandwritingRecog/data/Char_Recog/hhd_dataset/hdd_dataset/TRAIN/*/*.png")

####find standard deviation of new cropped dataset####
#std_x,std_y = findstd("HandwritingRecog/data/Char_Recog/hhd_dataset/hdd_dataset/TRAIN/*/*.png",avg_x,avg_y,img_size_all_x,img_size_all_y)

####create new dataset, pad according to max dimensions, save####
# CropAndPadding("C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog/data/Char_Recog/hhd_dataset/hhd_dataset/TRAIN/*/*.png")

####split all datasets into 80-20 train-test split####
#train_test_split("data/Char_Recog/cropped-padded-img-data/","data/binarized_monkbrill_split_40x40")sa
#dataset_split("C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog/data/Char_Recog/cropped-padded-img-data/")
#dataset_split('/home/jan/PycharmProjects/HandwritingRecog/data/Char_Recog/binarized_monkbrill_40x40')