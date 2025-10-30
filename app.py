import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# تحميل النموذج


import gdown

# تحميل النموذج تلقائيًا من Google Drive عند التشغيل
if not os.path.exists(MODEL_PATH):
    file_id = "1AbCdEfGhijkLMNOP"  # 🔹 استبدل هذا بالـ File ID الخاص بك
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# تحميل النموذج
model = tf.keras.models.load_model(MODEL_PATH)


# أسماء الفئات (حسب تدريبك)
class_names = ['تفاح 🍎', 'موز 🍌']

# واجهة Streamlit
st.set_page_config(page_title="تصنيف الفواكه 🍎🍌", page_icon="🍎", layout="centered")

st.title("🍎🔍 تطبيق تصنيف الفواكه بالذكاء الاصطناعي")
st.write("حمّل صورة لتتعرف على نوع الفاكهة 👇")

uploaded_file = st.file_uploader("📂 اختر صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 الصورة التي تم تحميلها", use_container_width=True)

    # تجهيز الصورة للنموذج
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # التنبؤ
    predictions = model.predict(img_array)
    confidence = predictions[0][0] if predictions.shape == (1, 1) else np.max(predictions)
    predicted_class = class_names[int(predictions[0][0] > 0.5)] if predictions.shape == (1, 1) else class_names[np.argmax(predictions)]
    confidence_percent = round(confidence * 100, 2)

    # عرض النتيجة
    if confidence_percent < 50:
        st.warning(f"❓ النتيجة غير مؤكدة!\n\nأقرب احتمال: **{predicted_class}** بنسبة ثقة {confidence_percent}%")
    else:
        st.success(f"🧠 التنبؤ: **{predicted_class}** 🎯\n\nنسبة الثقة: {confidence_percent}%")

    # رسم بياني بسيط للثقة
    import matplotlib.pyplot as plt

    probs = [predictions[0][0], 1 - predictions[0][0]] if predictions.shape == (1, 1) else predictions[0]
    fig, ax = plt.subplots()
    ax.bar(class_names, probs, color=["#FF9999", "#FFD966"])
    ax.set_ylabel("نسبة الثقة")
    ax.set_ylim([0, 1])
    st.pyplot(fig)
