import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import io

@st.cache_data
def load_data():
   return pd.read_csv("data_laptop.csv")

# --- SCORING FUNCTIONS ---
def get_cpu_score(cpu_name):
    """Assigns a score to a CPU based on its model."""
    cpu_name = str(cpu_name).lower()
    if 'intel celeron' in cpu_name: return 2
    if 'core i3' in cpu_name: return 4
    if 'core i5' in cpu_name: return 6
    if 'ryzen 5' in cpu_name: return 6
    if 'core i7' in cpu_name: return 8
    if 'ryzen 7' in cpu_name: return 8
    if 'core i9' in cpu_name: return 10
    if 'ryzen 9' in cpu_name: return 10
    return 1 # Default score for other CPUs

def get_gpu_score(gpu_name):
    """Assigns a score to a GPU based on its model."""
    gpu_name = str(gpu_name).lower()
    # --- UPDATED RULES ---
    if 'intel uhd' in gpu_name: return 2
    if 'ryzen integrated' in gpu_name: return 3
    # --- END UPDATED RULES ---
    if 'rtx 2050' in gpu_name: return 5
    if 'rtx 3050' in gpu_name: return 6
    if 'rtx 3060' in gpu_name: return 7
    if 'rtx 3070' in gpu_name: return 8
    if 'rtx 3080' in gpu_name: return 9
    if 'rtx 4070' in gpu_name: return 10
    return 1 # Default score for integrated/unknown GPUs

# --- MODEL TRAINING ---
@st.cache_data
def train_model(df):
    """Trains the Decision Tree model based on performance scores."""
    if df is None:
        return None, None

    # Create a copy to avoid modifying the cached original dataframe
    df_processed = df.copy()

    # Apply scoring
    df_processed['cpu_score'] = df_processed['CPU'].apply(get_cpu_score)
    df_processed['gpu_score'] = df_processed['GPU'].apply(get_gpu_score)
    df_processed['ram_score'] = df_processed['RAM'].apply(lambda x: x * 0.5)
    df_processed['storage_score'] = df_processed['Storage'].apply(lambda x: x * 0.02 * 1.5)

    # Performance score based purely on specs
    df_processed['performance_score'] = (
        df_processed['cpu_score'] + df_processed['gpu_score'] +
        df_processed['ram_score'] + df_processed['storage_score']
    )

    # Set threshold at the 75th percentile (top 25%) to align with UI text
    threshold = df_processed['performance_score'].quantile(0.4)
    df_processed['is_worth_it'] = df_processed['performance_score'] >= threshold

    # Features for the model
    features = ['cpu_score', 'gpu_score', 'ram_score', 'storage_score']
    X = df_processed[features]
    y = df_processed['is_worth_it']

    model = DecisionTreeClassifier(
        random_state=42, max_depth=5, min_samples_leaf=5
    )
    model.fit(X, y)

    return model, threshold

# --- USER INTERFACE (UI) ---
st.set_page_config(page_title="Prediksi Laptop 2025", layout="wide")
st.title("Sistem Prediksi Kelayakan Laptop")
st.markdown(
    "Aplikasi ini menilai kelayakan sebuah laptop"
)

st.divider()

df = load_data()
model, threshold = train_model(df)

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:

    cpu_options = [
        'Intel Celeron', 'Intel Core i3', 'Intel Core i5', 'AMD Ryzen 5',
        'Intel Core i7', 'AMD Ryzen 7', 'Intel Core i9', 'AMD Ryzen 9'
    ]
    selected_cpu = st.selectbox("Prosesor (CPU)", options=cpu_options, index=3)

    # --- UPDATED GPU OPTIONS ---
    gpu_options = [
        'Intel UHD',
        'Ryzen Integrated',
        'NVIDIA GeForce RTX 2050',
        'NVIDIA GeForce RTX 3050',
        'NVIDIA GeForce RTX 3060',
        'NVIDIA GeForce RTX 3070',
        'NVIDIA GeForce RTX 3080',
        'NVIDIA GeForce RTX 4070'
    ]
    selected_gpu = st.selectbox("Kartu Grafis (GPU)", options=gpu_options, index=2)

    selected_ram = st.selectbox("RAM (GB)", options=[4, 8, 16, 32, 64], index=2)

    selected_storage = st.number_input(
        "Peyimpanan (GB)",
        min_value=128, max_value=4096, value=512, step=128
    )

    predict_button = st.button("🔍 Prediksi Kelayakan", type="primary", use_container_width=True)

with col2:
    st.header("Hasil Prediksi")

    if predict_button:
        if model is not None and threshold is not None:
            # Calculate scores from user input
            input_cpu_score = get_cpu_score(selected_cpu)
            input_gpu_score = get_gpu_score(selected_gpu)
            input_ram_score = selected_ram * 0.5
            input_storage_score = selected_storage * 0.02 * 1.5
            input_performance_score = (
                input_cpu_score + input_gpu_score + input_ram_score + input_storage_score
            )

            # Create dataframe for prediction
            input_df = pd.DataFrame([[
                input_cpu_score, input_gpu_score, input_ram_score, input_storage_score
            ]], columns=['cpu_score', 'gpu_score', 'ram_score', 'storage_score'])

            # Make prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            # Display result
            if prediction[0]:
                st.success("✅ **Layak / Worth It (Performa di atas rata-rata)**")
                st.write(f"Tingkat Keyakinan: **{prediction_proba[0][1]*100:.2f}%**")
            else:
                st.error("❌ **Kurang Layak / Not Worth It (Performa di bawah standar)**")
                st.write(f"Laptop ini **tidak masuk** dalam kategori 25% teratas berdasarkan standar performa dari dataset.")
                st.progress(prediction_proba[0][1])
                st.write(f"Tingkat Keyakinan 'Layak': **{prediction_proba[0][1]*100:.2f}%**")

            st.divider()
            st.subheader("Detail Skor Performa Input Anda")
            
            delta_value = input_performance_score - threshold
            st.metric(
                label="Skor Performa Kalkulasi",
                value=f"{input_performance_score:.2f}",
                help="Delta menunjukkan selisih skor Anda dengan skor minimum 'Layak'."
            )
            st.metric(label="Ambang Batas Minimum 'Layak'", value=f"{threshold:.2f}")

            # Display bar chart
            chart_data = pd.DataFrame({
                "Komponen": ["CPU", "GPU", "RAM", "Storage"],
                "Skor": [input_cpu_score, input_gpu_score, input_ram_score, input_storage_score]
            })
            st.bar_chart(chart_data.set_index("Komponen"))
        else:
            st.error("Model belum berhasil dilatih. Pastikan dataset sudah diunggah dengan benar.")

    else:
        st.info("Masukkan spesifikasi lalu klik tombol 'Prediksi Kelayakan'")
