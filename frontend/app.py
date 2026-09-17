import streamlit as st
import requests
import json

API_BASE_URL = "http://127.0.0.1:5000"

st.set_page_config(
    page_title = "Face Recognization",
    page_icon="https://img.icons8.com/?size=100&id=pwjJUMU40zy8&format=png&color=000000",
    layout="wide"
)

st.title("Face Recognizaton Dashboard")
st.markdown("This app will recognize bollywood personalities")

with st.sidebar:
    st.header("Server config")
    base_url = st.text_input("Backend URL",value=API_BASE_URL)

    st.subheader("Diagnostics")

    if st.button("Check the API (GET /)",use_container_width=True):
        try:
            res = requests.get(f"{base_url}/",timeout=5)
            if res.status_code == 200:
                st.success(f"Status:Healthy:({res.status_code})")
                st.json(res.json())
            else:
                st.error("Status code:{res.status_code}")
                st.write(res.text)
        except requests.exceptions.ConnectionError:
            st.error("Flask server is not running")
        except Exception as e:
            st.error(f"error:{str(e)}")

    st.markdown("----------")

    if st.button("Run the server default image test (GET /predictor)",use_container_width=True):
            try:
                res = requests.get(f"{base_url}/predictor",timeout=15)
                if res.status_code == 200:
                    st.success(f"Default image test executed successfully")
                    try:
                        st.json(res.json())
                    except Exception:
                        st.write(res.text)
                else:
                    st.error("Status code:{res.status_code}")
                    st.write(res.text)
            except requests.exceptions.ConnectionError:
                st.error("Flask server is not running")
            except Exception as e:
                st.error(f"error:{str(e)}")

st.subheader("Face Recognization Inference")

input_mode = st.radio("choose input method",["File Uploader","Take a photo(Webcam)"],horizontal=True)

uploaded_file = None

if input_mode == "File Uploader":
    uploaded_file = st.file_uploader("Upload a image",type=["jpg","jpeg","png"])
else:
    uploaded_file = st.camera_input("capture a photo")

col1, col2 = st.columns([1,1],gap="large")

with col1:
    if uploaded_file is not None:
        st.image(uploaded_file,caption="Selcted Image",use_container_width=True)
    else:
        st.info("Please upload or capture image to proceed")

with col2:
    if uploaded_file is not None:
        st.markdown("Process Image")
        if st.button("Run recognization (POST /postpred)", type="primary", use_container_width=True):
            with st.spinner("Analyzing face..."):
                try:
                    files = {
                        "file":(uploaded_file.name, uploaded_file.getvalue(),uploaded_file.type)
                    }
                    response = requests.post(f"{base_url}/postpred",files=files,timeout=20)
                    if response.status_code ==  200:
                        st.success("Face Recognization completed!")
                        st.markdown("Result output")
                        st.json(response.json())
                    else:
                        st.error(f"Server returns HTTP {response.status_code}")
                        try:
                            st.json(response.json())
                        except Exception:
                            st.code(response.text)
                except requests.exceptions.ConnectionError:
                    st.error("Flask server is not running")
                except Exception as e:
                    st.error(f"error:{str(e)}")