import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# 1. Cấu hình Trang Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Salary Smart - Dự đoán Lương AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm CSS tùy chỉnh giao diện
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        text-align: center;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🤖 AI Salary Smart</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hệ thống dự đoán mức lương công việc AI bằng thuật toán Random Forest (Nhóm 7)</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Hàm Tải & Huấn luyện Mô hình (Caching để tối ưu tốc độ)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    train_file = "Nhom7_AIJobsSalary_train.xlsx"
    test_file = "Nhom7_AIJobsSalary_test.xlsx"

    # Kiểm tra sự tồn tại của file dữ liệu
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        return None, None, None, None, None, None, None

    # Đọc dữ liệu Excel
    df_train = pd.read_excel(train_file)
    df_test = pd.read_excel(test_file)

    # 1. Tự động xác định cột mục tiêu (Target Column) như trong Notebook
    salary_cols = [col for col in df_train.columns if any(k in col.lower() for k in ["salary", "luong", "wage", "pay"])]
    if len(salary_cols) > 0:
        usd_cols = [c for c in salary_cols if 'usd' in c.lower()]
        target_column = usd_cols[0] if usd_cols else salary_cols[0]
    else:
        num_cols = df_train.select_dtypes(include=[np.number]).columns
        target_column = num_cols[-1]

    # 2. Tách biến Features & Target
    features = [col for col in df_train.columns if col != target_column]
    X_train = df_train[features]
    y_train = df_train[target_column]
    X_test = df_test[features]
    y_test = df_test[target_column]

    # 3. Phân loại thuộc tính số và phân loại
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=[object, 'category']).columns.tolist()

    # 4. Xây dựng Pipeline tiền xử lý & Model
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    pipeline_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Huấn luyện mô hình
    pipeline_model.fit(X_train, y_train)

    # Dự đoán trên tập Test
    y_pred_all = pipeline_model.predict(X_test)

    # Tính toán chỉ số đánh giá
    mae = mean_absolute_error(y_test, y_pred_all)
    mse = mean_squared_error(y_test, y_pred_all)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_all)

    metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

    return pipeline_model, df_train, df_test, X_test, y_test, y_pred_all, metrics

# Gọi hàm load dữ liệu
with st.spinner('Đang tải dữ liệu và huấn luyện mô hình Random Forest...'):
    pipeline_model, df_train, df_test, X_test, y_test, y_pred_all, metrics = load_and_train_model()

if pipeline_model is None:
    st.error("⚠️ Không tìm thấy hai file dữ liệu `Nhom7_AIJobsSalary_train.xlsx` và `Nhom7_AIJobsSalary_test.xlsx`. Vui lòng đặt các file dữ liệu vào cùng thư mục với `app.py`!")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Sidebar Navigation Menu
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/illustrations/100/brain.png", width=80)
st.sidebar.title("Điều Hướng")
menu = st.sidebar.radio("Chọn chức năng:", ["🔮 Dự đoán Mức lương", "📊 Khám phá & Trực quan dữ liệu"])

# -----------------------------------------------------------------------------
# TAB 1: DỰ ĐOÁN MỨC LƯƠNG
# -----------------------------------------------------------------------------
if menu == "🔮 Dự đoán Mức lương":
    st.subheader("📋 Nhập thông tin công việc để dự đoán")

    # Tách danh sách cột số & phân loại từ tập TRAIN
    features = [c for c in df_train.columns if c in X_test.columns]
    num_cols = df_train[features].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_train[features].select_dtypes(include=[object, 'category']).columns.tolist()

    input_data = {}

    col1, col2, col3 = st.columns(3)

    # Chia các trường dữ liệu thành 3 cột giao diện
    all_cols = num_cols + cat_cols
    for idx, col_name in enumerate(all_cols):
        target_col = col1 if idx % 3 == 0 else (col2 if idx % 3 == 1 else col3)

        with target_col:
            # Tạo tên label dễ đọc
            display_label = col_name.replace('_', ' ').title()

            if col_name in num_cols:
                min_val = float(df_train[col_name].min())
                max_val = float(df_train[col_name].max())
                mean_val = float(df_train[col_name].median())
                
                # Nếu là số năm kinh nghiệm hoặc giá trị nguyên
                if df_train[col_name].dtype == 'int64':
                    input_data[col_name] = st.slider(f"⚙️ {display_label}", int(min_val), int(max_val), int(mean_val))
                else:
                    input_data[col_name] = st.number_input(f"⚙️ {display_label}", value=mean_val)
            else:
                unique_options = sorted(list(df_train[col_name].dropna().unique()))
                input_data[col_name] = st.selectbox(f"📌 {display_label}", options=unique_options)

    st.markdown("---")
    
    # Nút Bấm Dự Đoán
    if st.button("🚀 Dự đoán mức lương", type="primary", use_container_width=True):
        input_df = pd.DataFrame([input_data])
        predicted_salary = pipeline_model.predict(input_df)[0]
        
        mae_val = metrics['MAE']
        lower_bound = max(0, predicted_salary - mae_val)
        upper_bound = predicted_salary + mae_val

        st.balloons()
        st.success("✅ Dự đoán thành công!")
        
        # Hiển thị kết quả nổi bật
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.metric(
                label="💰 Mức lương dự đoán trung bình (USD/năm)",
                value=f"${predicted_salary:,.2f}"
            )
        with res_col2:
            st.metric(
                label="📊 Khoảng ước lượng tham khảo (± MAE)",
                value=f"${lower_bound:,.2f} - ${upper_bound:,.2f}"
            )

        st.info(f"💡 **Ghi chú học thuật:** Mô hình có sai số tuyệt đối trung bình (MAE) là **${mae_val:,.2f} USD**. Khoảng lương dự đoán thực tế nằm trong khung tin cậy dựa trên chỉ số đánh giá của tập TEST.")

# -----------------------------------------------------------------------------
# TAB 2: KHÁM PHÁ & TRỰC QUAN HÓA DỮ LIỆU
# -----------------------------------------------------------------------------
elif menu == "📊 Khám phá & Trực quan dữ liệu":
    st.subheader("📈 Đánh giá Hiệu suất Mô hình & Dữ liệu")

    # 1. Bảng Chỉ số Đánh giá
    st.markdown("### 1. Bảng chỉ số đánh giá hiệu suất (Tập TEST)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE (Sai số tuyệt đối)", f"{metrics['MAE']:,.2f}")
    m2.metric("RMSE (Căn bình phương)", f"{metrics['RMSE']:,.2f}")
    m3.metric("R² Score (Độ giải thích)", f"{metrics['R2']:.4f} ({metrics['R2']*100:.2f}%)")
    m4.metric("MSE", f"{metrics['MSE']:,.0f}")

    st.markdown("---")

    # 2. Xem dữ liệu mẫu từ Tập Test
    st.markdown("### 2. Xem dữ liệu mẫu từ tập TEST (Nhom7_AIJobsSalary_test.xlsx)")
    st.dataframe(df_test.head(10), use_container_width=True)

    st.markdown("---")

    # 3. Trực quan hóa Biểu đồ Báo cáo
    st.markdown("### 3. Biểu đồ đánh giá học thuật từ mô hình Random Forest")
    
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Biểu đồ 1: Line plot so sánh Thực tế vs Dự đoán (50 mẫu đầu)
    num_samples = min(50, len(y_test))
    sns.lineplot(ax=axes[0], data=y_test.values[:num_samples], label="Thực tế (Actual)", color='#2ca02c', marker='o', linewidth=2)
    sns.lineplot(ax=axes[0], data=y_pred_all[:num_samples], label="Dự đoán (Predicted)", color='#d62728', marker='x', linestyle='--', linewidth=2)
    axes[0].set_title(f'So sánh Thực tế vs Dự đoán ({num_samples} dòng đầu)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Chỉ mục mẫu (Index)', fontsize=11)
    axes[0].set_ylabel('Mức lương (USD/năm)', fontsize=11)
    axes[0].legend()

    # Biểu đồ 2: Scatter plot Actual vs Predicted
    sns.scatterplot(ax=axes[1], x=y_test.values, y=y_pred_all, alpha=0.7, color="#1f77b4", edgecolor='w', s=60)
    max_val = max(y_test.max(), y_pred_all.max())
    min_val = min(y_test.min(), y_pred_all.min())
    axes[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Đường hoàn hảo y = x')
    axes[1].set_title('Biểu đồ Tương quan: Actual vs Predicted', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Mức lương thực tế (Actual)', fontsize=11)
    axes[1].set_ylabel('Mức lương dự đoán (Predicted)', fontsize=11)
    axes[1].legend()

    # Biểu đồ 3: Biểu đồ Phân bố Sai số (Residuals)
    residuals = y_test.values - y_pred_all
    sns.histplot(ax=axes[2], x=residuals, kde=True, color="#9467bd", bins=20)
    axes[2].set_title('Biểu đồ Phân bố Sai số (Residuals)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Sai số dự đoán (Thực tế - Dự đoán)', fontsize=11)
    axes[2].set_ylabel('Mật độ (Density)', fontsize=11)

    plt.tight_layout()
    st.pyplot(fig)