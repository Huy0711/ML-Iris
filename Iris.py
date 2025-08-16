import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Đọc dữ liệu
df = pd.read_csv("Iris.csv")

# 2. Chuẩn bị dữ liệu
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# 3. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Huấn luyện mô hình K-NN
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 5. Đánh giá mô hình
y_pred = knn.predict(X_test)
print("🎯 Độ chính xác (Accuracy):", accuracy_score(y_test, y_pred))
print("\n📊 Ma trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred))
print("\n📄 Báo cáo phân loại:")
print(classification_report(y_test, y_pred))

# 6. Nhập thông số từ người dùng để dự đoán
print("\n🔍 Dự đoán loài hoa Iris từ thông số bạn nhập:")
sl = float(input("Nhập Sepal Length (cm): "))
sw = float(input("Nhập Sepal Width (cm): "))
pl = float(input("Nhập Petal Length (cm): "))
pw = float(input("Nhập Petal Width (cm): "))

# 7. Dự đoán loài
sample = [[sl, sw, pl, pw]]
pred_species = knn.predict(sample)
print(f"🌸 Loài hoa dự đoán: {pred_species[0]}")
