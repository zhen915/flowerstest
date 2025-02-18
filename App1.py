import streamlit as st
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO('best (9).pt')

plant_info = {
    "Adenium_obesum": {
        "中文名稱": "沙漠玫瑰(別名:富貴花)",
        "屬性": "夾竹桃科沙漠玫瑰屬",
        "花語": "忠貞不渝",
        "生長季節": "春季、夏季、秋季",
        "分佈": "栽植，外來種：原生於非洲東部到西亞的乾旱地中。",
        "毒性警告": "全株均有毒。家畜或幼兒誤食莖葉或乳汁，會引起心跳加速、心律不整等心臟疾病。",
        "食用安全性": "不可食用。",
        "藥用價值": "傷口敷料、治療潰瘍以及作為心臟強壯劑和治療性病的補助治療。",
        "環保資訊": "有助於節約水資源，通過吸收污染物並釋放氧氣來改善室內空氣質量。",
    },
}

st.set_page_config(page_title="花草辨識小助理", page_icon="🌱", layout="wide")

st.title("🌿 花草辨識小助理")
st.write("探索校園的植物世界，學習植物知識，感受自然之美！")

language = st.sidebar.selectbox("🌐 選擇語言", ("中文", "English"))
if language == "English":
    st.title("🌿 Plant Identification Application")
    st.write("Explore the plants around you, learn about their characteristics, and connect with nature!")
    upload_label = "Upload Image"
    take_photo_label = "Take Photo"
    result_label = "Recognition Results"
    history_label = "History"
    eco_info = "Environmental Benefits of Plants"
    eco_details = """
    Plants play a vital role in reducing global warming by providing oxygen and absorbing carbon dioxide, contributing to ecological balance.
    """
else:
    upload_label = "上傳圖片"
    take_photo_label = "拍照"
    result_label = "辨識結果"
    history_label = "歷史記錄"
    eco_info = "環保資訊"
    eco_details = """
    植物在減緩全球暖化中扮演重要角色，能提供氧氣並吸收二氧化碳，對維持生態平衡貢獻巨大。
    """


st.sidebar.header("功能選項")
option = st.sidebar.radio("選擇來源", (take_photo_label, upload_label))


if "history" not in st.session_state:
    st.session_state["history"] = []

image = None


if option == take_photo_label:
    st.subheader("📸 使用相機拍攝")
    camera_photo = st.camera_input("點擊下方按鈕拍照")
    if camera_photo is not None:
        image = Image.open(camera_photo)
        st.image(image, caption="拍攝的圖片", use_column_width=True)

elif option == upload_label:
    st.subheader("📤 上傳圖片")
    uploaded_file = st.file_uploader("選擇圖片", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上傳的圖片", use_column_width=True)


st.subheader(result_label)
if image is not None:
    image_np = np.array(image)
    results = model(image_np)
    predictions = results[0].boxes.data.cpu().numpy()

    if len(predictions) > 0:
        identified_plants = set()
        
        for row in predictions:
            plant_name = model.names[int(row[5])]  # 假設類別標籤在第六欄
            confidence = row[4]  # 假設信心分數在第五欄
            identified_plants.add((plant_name, confidence))

        
        for plant_name, confidence in identified_plants:
            st.markdown(f"**植物學名：{plant_name}** (信心分數：{confidence:.2f})")
            
            if plant_name in plant_info:
                info = plant_info[plant_name]
                st.write(f"🌸 中文名稱：{info['中文名稱']}")
                st.write(f"🌿 屬性：{info['屬性']}")
                st.write(f"💐 花語：{info['花語']}")
                st.write(f"🌱 生長季節：{info['生長季節']}")
                st.write(f"📍 分佈：{info['分佈']}")
                st.write(f"⚠️ 毒性警告：{info['毒性警告']}")
                st.write(f"🍴 食用安全性：{info['食用安全性']}")
                st.write(f"💊 藥用價值：{info['藥用價值']}")
                
        st.session_state["history"].append((image, identified_plants))
    else:
        st.write("未能辨識出植物")
else:
    st.write("請先上傳或拍攝圖片")


st.sidebar.subheader(history_label)
if len(st.session_state["history"]) > 0:
    for idx, (img, plants) in enumerate(st.session_state["history"]):
        with st.sidebar.expander(f"記錄 {idx + 1}"):
            st.image(img, caption="辨識圖片", use_column_width=True)
            for plant, _ in plants:
                st.write(f"- {plant}")
else:
    st.sidebar.write("目前尚無辨識記錄")


st.subheader(eco_info)


st.markdown("---")
st.markdown("✨ **探索自然，從身邊的植物開始！**")
