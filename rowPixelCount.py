import matplotlib.pyplot as plt
import cv2

img_path = 'data/image-data/binary/P21-Fg006-R-C01-R01-binarized.jpg'
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

hist = []
row_len = image.shape[1]
for row in image:
    hist.append(row_len-len(row.nonzero()[0]))
figure = plt.figure(figsize=(16, 12))
fs = 25
plt.plot(hist)
plt.ylim(0, max(hist)*1.1)
plt.xlabel("Row", fontsize=fs)
plt.ylabel("Black pixels", fontsize=fs)
plt.title("Binary image black pixel counting result", fontsize=fs)
plt.yticks(fontsize=fs-5)
plt.xticks(fontsize=fs-5)
plt.show()
figure.savefig("pixelCount.jpg")
figure.clf()
