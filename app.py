import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cấu hình giao diện trang web
st.set_page_config(page_title="AI Salary Smart", page_icon="🤖", layout="wide")

st.title("🤖 AI Salary Smart - Dự Đoán Lương Ngành AI")
st.markdown("Ứng dụng hỗ trợ dự đoán mức lương hàng năm dựa trên thuật toán **Random Forest Regression**.")

# Hàm tải và huấn luyện mô hình (sử dụng cache để tối ưu tốc độ)
@st.cache_resource
def load_model_and_data():
    # Đọc dữ liệu từ file Excel (đặt cùng thư mục)
    df_train = pd.read_excel("Nhom7_AIJobsSalary_train.xlsx")
    df_test = pd.read_excel("Nhom7_AIJobsSalary_test.xlsx")
    
    # Tự động xác định cột mục tiêu (lương)
    salary_cols = [col for col in df_train.columns if any(k in col.lower() for k in ["salary", "luong", "wage", "pay"])]
    if usd_cols := [c for c in salary_cols if 'usd' in c.lower()]:
        target_col = usd_cols[0]
    elif salary_cols:
        target_col = salary_cols[0]
    else:
        num_cols = df_train.select_dtypes(include=[np.number]).columns
        target_col = num_cols[-1]
        
    features = [col for col in df_train.columns if col != target_col]
    
    X_train = df_train[features]
    y_train = df_train[target_col]
    X_test = df_test[features]
    y_test = df_test[target_col]
    
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=[object, 'category']).columns.tolist()
    
    # Xây dựng Pipeline tiền xử lý và mô hình
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    model.fit(X_train, y_train)
    return model, df_train, df_test, X_test, y_test, target_col, features

try:
    model, df_train, df_test, X_test, y_test, target_col, features = load_model_and_data()
    
    # Tạo thanh menu bên trái
    menu = st.sidebar.selectbox("📂 Chọn chức năng", ["🔮 Dự đoán lương cá nhân", "📊 Khám phá & Trực quan hóa dữ liệu"])
    
    if menu == "🔮 Dự đoán lương cá nhân":
        st.subheader("📝 Nhập thông tin công việc ngành AI")
        
        user_input = {}
        col1, col2 = st.columns(2)
        
        for i, col in enumerate(features):
            with (col1 if i % 2 == 0 else col2):
                unique_vals = df_train[col].dropna().unique()
                if df_train[col].dtype in [np.number]:
                    min_v = float(df_train[col].min())
                    max_v = float(df_train[col].max())
                    mean_v = float(df_train[col].mean())
                    user_input[col] = st.number_input(f"{col}", min_value=min_v, max_value=max_v, value=mean_v)
                else:
                    options = list(unique_vals)
                    user_input[col] = st.selectbox(f"{col}", options)
        
        if st.button("🚀 Dự đoán mức lương", type="primary"):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            
            st.success(f"### 💰 Mức lương dự đoán: **${prediction:,.2f} USD/năm**")
            st.info("ℹ️ *Khoảng sai số tham khảo dựa trên sai số tuyệt đối trung bình (MAE) thực tế khoảng $11,441.70 USD.*")

    elif menu == "📊 Khám phá & Trực quan hóa dữ liệu":
        st.subheader("📈 Đánh giá hiệu suất mô hình trên tập Test")
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (Sai số tuyệt đối)", f"${mae:,.2f}")
        m2.metric("RMSE (Sai số căn bậc hai)", f"${rmse:,.2f}")
        m3.metric("R² Score (Hệ số xác định)", f"{r2:.4f} ({r2*100:.2f}%)")
        
        st.markdown("---")
        st.subheader("📊 Biểu đồ trực quan học thuật")
        
        # Vẽ 3 biểu đồ giống báo cáo
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Line plot so sánh
        n_samples = min(50, len(y_test))
        axes[0].plot(y_test.values[:n_samples], label="Thực tế (Actual)", color="#2ca02c", marker='o', linewidth=2)
        axes[0].plot(y_pred[:n_samples], label="Dự đoán (Predicted)", color="#d62728", marker='x', linestyle='--', linewidth=2)
        axes[0].set_title(f"So sánh Thực tế vs Dự đoán ({n_samples} mẫu đầu)")
        axes[0].set_xlabel("Chỉ mục mẫu (Index)")
        axes[0].set_ylabel("Mức lương (USD/năm)")
        axes[0].legend()
        
        # 2. Scatter plot tương quan
        axes[1].scatter(y_test.values, y_pred, alpha=0.7, color="#1f77b4", edgecolors='w')
        min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        axes[1].plot([min_v, max_v], [min_v, max_v], color='red', linestyle='--', linewidth=2, label="Đường hoàn hảo y = x")
        axes[1].set_title("Biểu đồ Tương quan: Actual vs Predicted")
        axes[1].set_xlabel("Mức lương thực tế")
        axes[1].set_ylabel("Mức lương dự đoán")
        axes[1].legend()
        
        # 3. Histogram phân bố sai số (Residuals)
        residuals = y_test.values - y_pred
        sns.histplot(residuals, kde=True, ax=axes[2], color="purple")
        axes[2].set_title("Biểu đồ Phân bố Sai số (Residuals)")
        axes[2].set_xlabel("Sai số (Thực tế - Dự đoán)")
        axes[2].set_ylabel("Mật độ (Density)")
        
        plt.tight_layout()
        st.pyplot(fig)
        
except Exception as e:
    st.error(f"⚠️ Lỗi khi tải file dữ liệu hoặc chạy mô hình: {e}")
    st.info("💡 Hãy đảm bảo các file `Nhom7_AIJobsSalary_train.xlsx` và `Nhom7_AIJobsSalary_test.xlsx` đã được đặt chung thư mục với file `app.py` này.")