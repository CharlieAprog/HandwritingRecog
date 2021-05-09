import matplotlib.pyplot as plt
import cv2
import numpy as np

img_path = 'data/image-data/Test/histogramTest.jpg'
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

hist = []
row_len = image.shape[1]
for row in image:
    hist.append(row_len - len(row.nonzero()[0]))

temp = []
thr = {}
c = 0
thr_num = 40
for idx, p in enumerate(hist):
    if p >= thr_num and hist[idx - 1] > thr_num and idx > 0:
        temp.append(p)
        c += 1
    elif len(temp) > 0:
        thr.setdefault(idx - c, temp)
        temp = []
        c = 0

line_heights = []
thr_peaks = {}
for idx, p in enumerate(thr.items()):
    line_heights.append(p[0] + len(p[1]) - p[0])

    thr_peaks[idx] = {
        "loc": [p[0], p[0] + len(p[1])],
        "value": max(p[1]),
        "lh": p[0] + len(p[1]) - p[0]
    }

avg_lh = sum(line_heights) / len(line_heights)
q3, q1 = np.percentile(line_heights, [75, 25])
iqr = q3-q1
outlier = q1-1.5*iqr

# ----------------------- Histogram plotting -----------------------
# figure = plt.figure(figsize=(16, 12))
# fs = 25
# plt.plot(hist)
# plt.ylim(0, max(hist)*1.1)
# plt.xlabel("Row", fontsize=fs)
# plt.ylabel("Black pixels", fontsize=fs)
# plt.title("Binary image black pixel counting result", fontsize=fs)
# plt.yticks(fontsize=fs-5)
# plt.xticks(fontsize=fs-5)
# # plt.grid()
# plt.show()
# ----------------------- Histogram plotting -----------------------

figure = plt.figure(figsize=(16, 12))
fs = 25
plt.imshow(image)
line_heights = []
for k in thr_peaks.keys():
    for _ in thr_peaks[k].keys():
        if thr_peaks[k]["lh"] <= outlier:
            pass
        else:
            for idx, loc in enumerate(thr_peaks[k]["loc"]):
                if idx == 0:
                    plt.axhline(y=loc - avg_lh // 3, color="r", linestyle="-")
                else:
                    plt.axhline(y=loc + avg_lh // 3, color="r", linestyle="-")
plt.show()
