from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from tinydb import TinyDB, Query
import os
import cv2
import face_recognition
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import shutil
from pydub import AudioSegment
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastdtw import fastdtw
import json


# Set paths for FFmpeg
AudioSegment.converter = "D:/ffmpeg-7.1-essentials_build/bin/ffmpeg.exe"
AudioSegment.ffprobe = "D:/ffmpeg-7.1-essentials_build/bin/ffprobe.exe"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


USER_VIDEOS_DIR = "user_videos"
os.makedirs(USER_VIDEOS_DIR, exist_ok=True)

db = TinyDB("database.json")

dashboard_data = {}

class KeystrokeData(BaseModel):
    key_timings: list[float]

def save_video_file(username: str, video_file: UploadFile):
    file_path = os.path.join(USER_VIDEOS_DIR, f"{username}.webm")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)
    return file_path

def extract_face_from_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            print(f"Face detected at: {face_locations}")  # Debugging log
            video_capture.release()
            return face_recognition.face_encodings(frame, known_face_locations=[face_locations[0]])[0]
    video_capture.release()
    print("No face detected in video")  # Debugging log
    return None


def compare_faces(registered_video_path, login_video_path):
    registered_face_encoding = extract_face_from_video(registered_video_path)
    login_face_encoding = extract_face_from_video(login_video_path)
    if registered_face_encoding is None or login_face_encoding is None:
        return False
    # Convert numpy.bool_ to native bool
    return bool(face_recognition.compare_faces([registered_face_encoding], login_face_encoding, tolerance=0.5)[0])

def extract_audio_from_video(video_path):
    try:
        audio = AudioSegment.from_file(video_path)
        wav_path = video_path.replace(".webm", ".wav")
        audio.export(wav_path, format="wav")
        print(f"Extracted audio saved at: {wav_path}")  # Debugging log
        return wav_path
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return None


def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

def compare_audio(registered_video_path, login_video_path):
    reg_wav_path = extract_audio_from_video(registered_video_path)
    login_wav_path = extract_audio_from_video(login_video_path)
    reg_audio_features = extract_audio_features(reg_wav_path)
    login_audio_features = extract_audio_features(login_wav_path)
    os.remove(reg_wav_path)
    os.remove(login_wav_path)
    return 1 - cosine(reg_audio_features, login_audio_features) > 0.5

def compare_key(stored_template, login_template):
    print(f"Stored Keystrokes: {stored_template}")  # Debugging
    print(f"Login Keystrokes: {login_template}")  # Debugging
    distance, _ = fastdtw(stored_template, login_template)
    print(f"Keystroke Distance: {distance}")  # Debugging
    return distance < 400.0  # Adjusted from 150.0


@app.post("/signup")
async def signup(username: str = Form(...), password: str = Form(...), video: UploadFile = File(...), keystroke_timings: str = Form(...)):
    print(f"Signup attempt for: {username}")  # Debugging
    
    # Check if the username already exists
    if db.search(Query().uname == username):
        print("Username already exists.")  # Debugging
        return JSONResponse(content={"success": False, "error": "Username already exists."}, status_code=400)

    try:
        # Save video file
        video_file_path = save_video_file(username, video)
        keystroke_data = json.loads(keystroke_timings)

        # Insert user data into the database
        db.insert({'uname': username, 'password': password, 'video_file': video_file_path, 'keystroke_timings': keystroke_data})

        print("Signup successful")  # Debugging
        return JSONResponse(content={"success": True, "message": "Signup successful"}, status_code=200)

    except Exception as e:
        print(f"Error during signup: {e}")  # Debugging
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), video: UploadFile = File(...), keystroke_timings: str = Form(...),cursor_movements: str = Form(...)):
    result = db.search(Query().uname == username)
    if not result:
        return JSONResponse(content={"success": False, "error": "User does not exist."}, status_code=404)
    if result[0]['password'] != password:
        return JSONResponse(content={"success": False, "error": "Invalid password."}, status_code=401)
    
    login_video_path = os.path.join(USER_VIDEOS_DIR, f"temp_{username}.webm")
    with open(login_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    registered_video_path = result[0]['video_file']
    is_same_face = compare_faces(registered_video_path, login_video_path)
    is_same_voice = compare_audio(registered_video_path, login_video_path)
    stored_template = result[0].get("keystroke_timings", [])
    login_template = json.loads(keystroke_timings)
    is_same_keystroke = compare_key(stored_template, login_template)
    
    os.remove(login_video_path)
    cursor_movements_list = json.loads(cursor_movements)
    cursor_movement_detected = len(cursor_movements_list) > 3
    if len( cursor_movements_list) > 3:
            print("More than 3 cursor movements detected.")
    else:
            print("3 or fewer cursor movements detected.")
    # Save authentication results in the database
    db.update({
    "auth_results": {
        "face_match": bool(is_same_face),
        "voice_match": bool(is_same_voice),
        "keystroke_match": bool(is_same_keystroke),
        "cursor_mov": cursor_movement_detected
    }
}, Query().uname == username)
    
    # Return the results in the response
    if not (is_same_face and is_same_voice and is_same_keystroke):
        return JSONResponse(content={
            "success": False,
            "error": "Authentication failed.",
            "face_match": bool(is_same_face),
            "voice_match": bool(is_same_voice),
            "keystroke_match": bool(is_same_keystroke),
             "cursor_mov": cursor_movement_detected
        }, status_code=401)
    
    return JSONResponse(content={
        "success": True,
        "message": "User authenticated successfully",
        "face_match": bool(is_same_face),
        "voice_match": bool(is_same_voice),
        "keystroke_match": bool(is_same_keystroke),
        "cursor_mov": cursor_movement_detected
    }, status_code=200)


@app.get("/dashboard/{username}")
async def get_dashboard(username: str):
    result = db.search(Query().uname == username)
    if not result or "auth_results" not in result[0]:
        return JSONResponse(content={"error": "User data not found."}, status_code=404)
    
    return JSONResponse(content=result[0]["auth_results"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)