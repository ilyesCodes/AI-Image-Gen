import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from diffusers import StableDiffusionPipeline
import torch
import time
import sqlite3
import hashlib
from flask import (
    Flask, request, session, redirect, url_for, render_template, send_from_directory, flash
)
from datetime import datetime
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()
# Init Server App
app = Flask(__name__)
app.secret_key = os.getenv("SESSION_SECRET")

@app.template_filter("datetimeformat")
def datetimeformat(value):
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")

#  Constants
DB = "users_images.db"
IMAGE_FOLDER= "generated_images"
os.makedirs(IMAGE_FOLDER,exist_ok=True)

# Load models once at startup
whisper_model = whisper.load_model("base")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

#  Database Helpers

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
def hash_password(password:str)->str:
    return hashlib.sha256(password.encode()).hexdigest()


def check_password(stored_hash:str,password:str)->bool:
    return stored_hash == hash_password(password)


#  Step One Record Audio
def record_audio(filename,duration=5, sample_rate=16000):
    print("Recording ... Speak Now!")
    audio = sd.rec(int(duration * sample_rate),samplerate=sample_rate,channels=1,dtype="int16")
    sd.wait()
    wav.write(filename,sample_rate,audio)
    print("Recording Finsihed!")

#  Step Two Transcribe Audio with Whisper
def transcribe_audio(filename):
    print("Transcribing audio")
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    text = result["text"]
    print(f'Transcription: {text}')
    return text

#  Step Three Generate Image From Text
def generate_image(prompt,user_id):
    print("Generating Image From Prompt...")
    image = pipe(prompt).images[0]
    timestamp = int(time.time())
    filename= f"user_{user_id}_{timestamp}.png"
    output_path = os.path.join(IMAGE_FOLDER,filename)
    image.save(output_path)
    print(f"Image Saved to {output_path}")
    return filename

# Routes

@app.route("/")
def home():
    if 'user_id' in session:
        return redirect(url_for("gallery"))
    return render_template("index.html")


@app.route("/signup",methods=["GET","POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            flash("Username and Password are required","error")
            return redirect(url_for("signup"))
        
        password_hash = hash_password(password)

        try:
            with get_db() as db:
                db.execute("INSERT INTO users (username,password_hash) VALUES (?,?)",(username,password_hash))
                db.commit()
                flash("User created, Please Login","success")
                return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exsits","error")
            return redirect(url_for("signup"))

    return render_template("signup.html")

@app.route("/login",methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            flash("Username and Password are required","error")
            return redirect(url_for("login"))
        
        with get_db() as db:
            user = db.execute("SELECT * FROM users WHERE username = ?",(username,)).fetchone()

        if user and check_password(user["password_hash"],password):
            session["user_id"] = user["id"]
            session["username"] = username
            flash("Logged In Successfully","success")
            return redirect(url_for("gallery"))
        else:
            flash("Invalid username or password","error")
            return redirect(url_for("login"))
            
        
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out","success")
    return redirect(url_for("home"))

@app.route("/delete",methods=["POST"])
def delete_image():
    filename = request.form.get("filename")
    if not filename:
        flash("No Filename Provided","danger")
        return redirect(url_for("gallery"))

    image_path = os.path.join(IMAGE_FOLDER,filename)


    try:
        with get_db() as db:
            db.execute("DELETE FROM images WHERE filename = ?",(filename,))
            db.commit()

        if os.path.exists(image_path):
            os.remove(image_path)
            flash("Image deleted successfully","success")
        else:
            flash("Image Not Found ON Disk","error")
    except Exception as e:
            flash("500 Internal Server Error","error")
            print("Error in Image Deletetion Route: ",e)

    return redirect(url_for("gallery"))

@app.route("/generate",methods=["POST"])
def generate():
    if "user_id" not in session:
        flash("Please login first","error")
        return redirect(url_for("login"))

    user_id = session["user_id"]

    with tempfile.NamedTemporaryFile(suffix="wav",delete=False) as tmpfile:
        record_audio(tmpfile.name,duration=5)

        prompt = transcribe_audio(tmpfile.name)

        filename = generate_image(prompt,user_id)

        with get_db() as db:
            db.execute("INSERT INTO images (user_id,filename, timestamp) VALUES (?, ?, ?)",(user_id,filename,int(time.time())))
            db.commit()
        os.unlink(tmpfile.name)
    
    flash(f"Image generated for prompt: {prompt}","success")

    return redirect(url_for("gallery"))


@app.route("/gallery")
def gallery():
    if "user_id" not in session:
        flash("Please Login First","error")
        return redirect(url_for("login"))
    
    user_id = session["user_id"]
    with get_db() as db:
        images = db.execute("SELECT filename, timestamp FROM images WHERE user_id = ? ORDER BY timestamp DESC",(user_id,)).fetchall()

    return render_template("gallery.html",images=images)

@app.route("/images/<filename>")
def serve_image(filename):
    if "user_id" not in session:
        return "Unauthorized", 401

    user_id = session["user_id"]
    with get_db() as db:
        image = db.execute("SELECT * FROM images WHERE filename = ? AND user_id = ?",(filename,user_id)).fetchone()

    if not image:
        return "Image Not Found or Access Denied", 404
    
    return send_from_directory(IMAGE_FOLDER,filename,as_attachment=True)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)