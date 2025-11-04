import streamlit as st
import numpy as np
from PIL import Image
import cv2
from scipy.stats import skew, kurtosis, entropy
from skimage.metrics import structural_similarity as ssim

# --- Fungsi Bantuan ---
# (Fungsi-fungsi ini SAMA PERSIS dengan kode sebelumnya)
# load_image, to_grayscale, calculate_single_image_stats, 
# calculate_comparison_stats

def load_image(uploaded_file):
    """Memuat gambar yang di-upload pengguna."""
    try:
        image = Image.open(uploaded_file)
        return np.array(image)
    except Exception as e:
        st.error(f"Error memuat gambar: {e}")
        return None

def to_grayscale(image_array):
    """Konversi gambar ke grayscale jika berwarna."""
    if image_array.ndim == 3:
        # Konversi ke Grayscale menggunakan OpenCV
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return image_array

def calculate_single_image_stats(image_gray):
    """Menghitung statistik untuk satu gambar: Entropy, Skewness, Kurtosis."""
    flat_data = image_gray.flatten()
    hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    hist_prob = hist.ravel() / hist.sum()
    hist_prob = hist_prob[hist_prob > 0]
    
    ent = entropy(hist_prob, base=2)
    sk = skew(flat_data)
    kur = kurtosis(flat_data)
    
    return ent, sk, kur

def calculate_comparison_stats(img1_gray, img2_gray):
    """Menghitung metrik perbandingan antara dua gambar."""
    height, width = img1_gray.shape
    img2_resized = cv2.resize(img2_gray, (width, height))
    
    flat1 = img1_gray.flatten()
    flat2 = img2_resized.flatten()
    pearson_corr = np.corrcoef(flat1, flat2)[0, 1]
    
    hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2_resized], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    chi_square_val = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    
    ssim_val = ssim(img1_gray, img2_resized, data_range=img1_gray.max() - img1_gray.min())
    
    return pearson_corr, chi_square_val, ssim_val

# --- Antarmuka Streamlit ---

st.set_page_config(page_title="Analisis Citra", layout="wide")
st.title("🔬 Aplikasi Analisis dan Perbandingan Citra")
st.write("Dibuat untuk Mata Kuliah Pengolahan Citra")

if 'img1_gray' not in st.session_state:
    st.session_state.img1_gray = None

col1, col2 = st.columns(2)

# --- KOLOM 1 ---
with col1:
    st.header("Citra Pertama")
    file1 = st.file_uploader("Upload Citra 1 (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"], key="file1")
    
    if file1:
        img1_arr = load_image(file1)
        if img1_arr is not None:
            st.image(img1_arr, caption="Citra 1 Asli", use_container_width=True)
            
            img1_gray = to_grayscale(img1_arr)
            st.session_state.img1_gray = img1_gray
            
            ent, sk, kur = calculate_single_image_stats(img1_gray)
            
            st.subheader("Statistik Citra 1")
            st.metric(label="Entropy", value=f"{ent:.4f}")
            st.metric(label="Skewness", value=f"{sk:.4f}")
            st.metric(label="Kurtosis", value=f"{kur:.4f}")

# --- KOLOM 2 ---
with col2:
    st.header("Citra Kedua")
    file2 = st.file_uploader("Upload Citra 2 (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"], key="file2")
    
    if file2:
        img2_arr = load_image(file2)
        if img2_arr is not None:
            st.image(img2_arr, caption="Citra 2 Asli", use_container_width=True)
            
            img2_gray = to_grayscale(img2_arr)
            
            # --- TAMBAHAN BARU ---
            # Hitung dan tampilkan statistik individu untuk Citra 2
            ent2, sk2, kur2 = calculate_single_image_stats(img2_gray)
            
            st.subheader("Statistik Citra 2")
            st.metric(label="Entropy", value=f"{ent2:.4f}")
            st.metric(label="Skewness", value=f"{sk2:.4f}")
            st.metric(label="Kurtosis", value=f"{kur2:.4f}")
            # --- AKHIR TAMBAHAN ---
            
            st.divider() # Tambahkan garis pemisah

            # Pastikan citra 1 sudah ada sebelum membandingkan
            if st.session_state.img1_gray is not None:
                
                # Hitung perbandingan
                pearson, chi, ssim_val = calculate_comparison_stats(st.session_state.img1_gray, img2_gray)
                
                st.subheader("Hasil Perbandingan (Citra 1 vs Citra 2)")
                
                st.metric(label="Pearson Correlation", value=f"{pearson:.4f}")
                st.metric(label="Chi-Square (Histogram)", value=f"{chi:.4f}")
                st.metric(label="Nilai Matching (SSIM)", value=f"{ssim_val:.4f}")

            else:
                st.warning("Harap upload Citra 1 terlebih dahulu untuk melakukan perbandingan.")