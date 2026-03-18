import urllib.request
from scipy.io import arff
import pandas as pd

url = "http://promise.site.uottawa.ca/SERepository/datasets/cocomo81.arff"
print("Đang kết nối tới kho lưu trữ PROMISE...")

try:
    # Tải file .arff trực tiếp từ máy chủ học thuật
    urllib.request.urlretrieve(url, "cocomo81.arff")
    print("[+] Tải thành công file gốc! Đang chuyển đổi định dạng...")
    
    # Đọc dữ liệu từ file arff
    data, meta = arff.loadarff("cocomo81.arff")
    df = pd.DataFrame(data)
    
    # Giải mã các dữ liệu dạng byte (b'text') thành chuỗi bình thường (text)
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
    # Làm sạch: xóa các dòng thiếu dữ liệu nếu có
    df = df.dropna()
        
    # Lưu thành file CSV
    df.to_csv("cocomo81.csv", index=False)
    print("[+] Xong! File 'cocomo81.csv' đã xuất hiện trong thư mục của bạn.")

except Exception as e:
    print("[-] Có lỗi xảy ra khi tải:", e)