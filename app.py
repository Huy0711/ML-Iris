from flask import Flask, render_template, request, redirect, url_for
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__)

# ---------- Chuẩn bị & huấn luyện model (k = 5) ----------
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names  # ['setosa' 'versicolor' 'virginica']

# chia train/test để hiển thị accuracy (không bắt buộc cho chức năng)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# tính accuracy cho thông tin (tùy chọn)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", accuracy=accuracy, k=k)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sl = float(request.form.get("sepal_length", "").strip())
        sw = float(request.form.get("sepal_width", "").strip())
        pl = float(request.form.get("petal_length", "").strip())
        pw = float(request.form.get("petal_width", "").strip())
    except ValueError:
        # Nếu dữ liệu không hợp lệ, quay về trang chính với thông báo (đơn giản)
        return redirect(url_for('index'))

    sample = np.array([[sl, sw, pl, pw]])
    pred = knn.predict(sample)[0]
    pred_name = target_names[pred]

    # Có thể trả thêm các thông số khác nếu muốn
    return render_template("result.html",
                           sepal_length=sl, sepal_width=sw,
                           petal_length=pl, petal_width=pw,
                           prediction=pred_name.capitalize())

if __name__ == "__main__":
    app.run(debug=True)
