import cv2
import numpy as np
import pandas as pd
import os
from skimage import measure

# 箱詰め法でフラクタル次元を計算する関数
def calculate_fractal_dimension(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
    def box_count(img, k):
        S = np.add.reduceat(
            np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
            np.arange(0, img.shape[1], k), axis=1)
        return np.count_nonzero(S)

    sizes = 2**np.arange(1, 8)
    counts = [box_count(binary_image, size) for size in sizes]
    
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

# 画像フォルダとExcelファイルの指定
image_folder = r"C:\Users\otake\fractal"
output_excel = r"C:\Users\otake\fractal\fractal.xlsx"

# フラクタル次元を計算し、データをリストに格納
data = []
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    fd = calculate_fractal_dimension(img_path)
    data.append([img_file, fd])

# DataFrameに変換してExcelに保存
df = pd.DataFrame(data, columns=['Image Name', 'Fractal Dimension'])
df.to_excel(output_excel, index=False)

print(f"Excelファイルが{output_excel}に保存されました。")
