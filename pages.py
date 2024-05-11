import cv2
import numpy as np
import av
import mediapipe as mp
import streamlit as st
import queue
import joblib
import pandas as pd
from statistics import mode
from typing import List, NamedTuple
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import streamlit.components.v1 as components
from gtts import gTTS
import os
import base64
import time

def load_model():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    model = joblib.load("model.pkl")

    return model, mp_drawing, mp_drawing_styles, mp_hands, hands

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )
        
class Detection(NamedTuple):
    label: str   

def realtime():
    def mapping_landmark(landmarks):
        return landmarks.x, landmarks.y, landmarks.z
    
    def process(image):
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if landmark:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
        try:
            list_result = list(np.array([mapping_landmark(landmarks) for landmarks in results.multi_hand_world_landmarks[0].landmark]).ravel())
        except:
            list_result = []

        np_array = np.array(list_result).reshape(1, -1)
        if np_array.shape[1] != 0:
            prediction = model.predict(np_array)[0].upper()
        else:
            prediction = None

        detections = Detection(label=prediction)
        result_queue.put(detections)

        return cv2.flip(image, 1)

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = process(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def update_prediction():
        while True:
            value = result_queue.get()[0]
            if value is None:
                value = ""
            st.columns(1)[0].markdown("<h1 style='text-align: center; color: red;'>{}</h1>".format(value), unsafe_allow_html=True)  
    
    st.markdown("<h1 style='text-align: center; color: white; font-size: 20px;'>Fingerspelling Recognition</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white; font-size: 20px;'>{}</h1>".format(os.listdir()), unsafe_allow_html=True)
    model, mp_drawing, mp_drawing_styles, mp_hands, hands = load_model()
    result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

    webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={#"video": {"frameRate": {"ideal": 15}}, 
                                  "video": True,
                                  "audio": False},
        video_html_attrs={"style": {"width": "70%", "margin": "0 auto", "border": "3px black solid"},
                          "controls": False,
                          "autoPlay": True},
        video_frame_callback=video_frame_callback,
        async_processing=True,
    )
    landmark = False
    if webrtc_ctx.state.playing:
        landmark = st.checkbox("Show the landmark", value=False)
        with st.empty():
            update_prediction()
                    
def games():
    def mapping_landmark(landmarks):
        return landmarks.x, landmarks.y, landmarks.z
    
    def process(image):
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if landmark:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
        try:
            list_result = list(np.array([mapping_landmark(landmarks) for landmarks in results.multi_hand_world_landmarks[0].landmark]).ravel())
        except:
            list_result = []

        np_array = np.array(list_result).reshape(1, -1)
        if np_array.shape[1] != 0:
            prediction = model.predict(np_array)[0].upper()
        else:
            prediction = None

        detections = Detection(label=prediction)
        result_queue.put(detections)

        return cv2.flip(image, 1)

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = process(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def update_prediction():
        value = np.nan
        list_value = []
        list_word = []
        i = 0
        j = 0
        frame_steps = 10
        th = 0.85
        for k in range(n_word):
            col[k].markdown("<h1 style='text-align: center; color: white;'>{}</h1>".format(str(user_input)[k].upper()), unsafe_allow_html=True)
        while True:
            if j < n_word:
                if i == frame_steps:
                    value = mode(list_value)
                    if value is not None and round(list_value.count(value) / len(list_value), 2) > th:
                        if value.upper() == str(user_input)[j].upper():
                            color = "green"
                        else:
                            color = "red"
                            value = "&#10006"
                            #value = "U+1F480"
                        col[j].markdown("<h1 style='text-align: center; color: {};'>{}</h1>".format(color,value), unsafe_allow_html=True)
                        j+=1
                        list_word.append(value)
                    list_value = []
                    i=0
                else:
                    list_value.append(result_queue.get()[0])
                    i+=1
            else:
                value = result_queue.get()[0]
                if "".join(list_word).upper() == str(user_input).upper():
                    gTTS(text="".join(list_word), lang="en", slow=False).save("sound.mp3")
                    autoplay_audio(os.getcwd() + "/sound.mp3")
                    time.sleep(3)
                    st.write("Congratulations")
                    break
                else:
                    st.write("Try Again")
    
    model, mp_drawing, mp_drawing_styles, mp_hands, hands = load_model()
    result_queue: "queue.Queue[List[Detection]]" = queue.Queue()
    st.markdown("<h1 style='text-align: center; color: white; font-size: 20px;'>Sign the Word</h1>", unsafe_allow_html=True)
            
    user_input = st.text_input("Input word")
    if str(user_input).isalpha():
        webrtc_ctx = webrtc_streamer(
            key="WYH",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={#"video": {"frameRate": {"ideal": 15}}, 
                                      "video": True,
                                      "audio": False},
            video_html_attrs={"style": {"width": "100%", "margin": "0 auto", "border": "3px black solid"},
                              "controls": False,
                              "autoPlay": True},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )

        n_word = len(user_input)
        row1 = st.columns(n_word)
        row2 = st.columns(n_word)
        col = row1 + row2
        landmark = False
        if webrtc_ctx.state.playing:
            landmark = st.checkbox("Show the landmark", value=False)
            with st.empty():
                update_prediction()
    else:
        st.write('Please type in a string ')
