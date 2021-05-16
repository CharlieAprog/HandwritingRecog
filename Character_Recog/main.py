from data_investigation import *

#Data investigation before crop
avg_x,avg_y = data_size_stats("../data/monkbrill/*/*.pgm")
#Get stats for cropped images
max_x,max_y, img_size_all_x, img_size_all_y = CroppedCharAnalysis("../data/monkbrill/*/*.pgm")
#find standard deviation of new cropped dataset
std_x,std_y = findstd("../data/monkbrill/*/*.pgm",avg_x,avg_y,img_size_all_x,img_size_all_y)
#create new dataset, pad according to max dimensions, save
CropAndPadding("../data/monkbrill/*/*.pgm")