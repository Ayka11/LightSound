import matplotlib
matplotlib.use('Agg') 
import colorsys
from flask import Flask, render_template, request,session
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import string
from docx import Document
import PyPDF2
import pdfplumber
import soundfile as sf
#import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from scipy.fft import fft
import io
import base64
import pandas as pd
from pydub import AudioSegment

from dash import Dash, dcc, html
import dash_core_components as dcc
import dash_html_components as html
from dash import Dash, dcc, html, callback, Input, Output
import plotly.graph_objs as go
import plotly.express as px
global session
app = Flask(__name__)
app.secret_key = 'lantop'  # Set a secret key for session



# Initialize Dash app
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')


Fs = 44100  # Sampling frequency

# Frequency ranges and names of piano notes
notes = ['A0', 'A#0/Bb0', 'B0', 'C1', 'C#1/Db1', 'D1', 'D#1/Eb1', 'E1', 'F1', 'F#1/Gb1', 'G1', 'G#1/Ab1', 'A1', 
         'A#1/Bb1', 'B1', 'C2', 'C#2/Db2', 'D2', 'D#2/Eb2', 'E2', 'F2', 'F#2/Gb2', 'G2', 'G#2/Ab2', 'A2', 
         'A#2/Bb2', 'B2', 'C3', 'C#3/Db3', 'D3', 'D#3/Eb3', 'E3', 'F3', 'F#3/Gb3', 'G3', 'G#3/Ab3', 'A3', 
         'A#3/Bb3', 'B3', 'C4', 'C#4/Db4', 'D4', 'D#4/Eb4', 'E4', 'F4', 'F#4/Gb4', 'G4', 'G#4/Ab4', 'A4', 
         'A#4/Bb4', 'B4', 'C5', 'C#5/Db5', 'D5', 'D#5/Eb5', 'E5', 'F5', 'F#5/Gb5', 'G5', 'G#5/Ab5', 'A5', 
         'A#5/Bb5', 'B5', 'C6', 'C#6/Db6', 'D6', 'D#6/Eb6', 'E6', 'F6', 'F#6/Gb6', 'G6', 'G#6/Ab6', 'A6', 
         'A#6/Bb6', 'B6', 'C7', 'C#7/Db7', 'D7', 'D#7/Eb7', 'E7', 'F7', 'F#7/Gb7', 'G7', 'G#7/Ab7', 'A7', 
         'A#7/Bb7', 'B7', 'C8']
freqs_org = [27.50, 29.14, 30.87, 32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.00, 51.91, 55.00, 58.27, 
         61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 
         130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 
         261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 
         523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 
         1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.00, 
         1864.66, 1975.53, 2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 
         3322.44, 3520.00, 3729.31, 3951.07, 4186.01]

# Set colors for each frequency
colors = [[139/255, 0, 0]] * len(freqs_org)

def read_docx(file):
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def read_pdf(file):
    full_text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            full_text.append(page.extract_text())
    return '\n'.join(full_text)

def split_image_into_chunks(image, chunk_size):
    width, height = image.size
    chunks = []
    for i in range(0, width, chunk_size):
        chunk = image.crop((i, 0, min(i + chunk_size, width), height))
        buf = io.BytesIO()
        chunk.save(buf, format='PNG')
        buf.seek(0)
        chunks.append(base64.b64encode(buf.getvalue()).decode())
    return chunks

def generate_color_palette():
    chars = string.digits + string.ascii_lowercase + string.ascii_uppercase + '!?., '
    palette = {}
    
    num_colors = len(chars)
    for i, char in enumerate(chars):
        hue = i**2 / num_colors   
        lightness = 0.3 + (i % 2) * 0.4 
        saturation = 0.8 + (i % 3) * 0.6 
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        rgb = tuple(int(255 * x) for x in rgb)
        palette[char] = rgb
    palette['A']  = (200,200,200)
    palette['b'] = (165,0,165)
    palette['c'] = (100,200,255)
    return palette

def char_to_color(char, palette):
    return palette.get(char, (100,100,100)) 

def color_to_char(color, palette):
    for char, col in palette.items():
        if col == color:
            return char
    return '?'

def string_to_color_pattern(input_string, palette, cell_width=200, cell_height=200):
    length = len(input_string)
    width = length * cell_width
    height = cell_height + cell_height // 2  
    image = Image.new("RGB", (width, height), (255, 255, 255))  
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=50)

    color_code = []
    for i, char in enumerate(input_string):
        color = char_to_color(char, palette)
        color_code.append(color)
        top_left = (i * cell_width, 0)
        bottom_right = ((i + 1) * cell_width, cell_height)
        draw.rectangle([top_left, bottom_right], fill=color)
        text_width, text_height = 20,20
        text_x = top_left[0] + (cell_width - text_width) / 2
        text_y = cell_height + (cell_height // 2 - text_height) / 2
        draw.text((text_x, text_y), char, fill=(0, 0, 0), font=font, stroke_width=1)
    
    return image, color_code

def color_code_to_string(color_code, palette):
    return ''.join(color_to_char(tuple(color), palette) for color in color_code)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    global session
    
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    if file:
        y, sr = librosa.load(file)

        
        #y = y.reshape(-1, 1)

        if y.ndim > 1 and y.shape[1] > 1:
            print("Audio has more than one channel. Using the first channel.")
            y = y[:, 0]  # Select the first channel
    

        
        D = np.abs(librosa.stft(y))
        D_db = librosa.amplitude_to_db(D, ref=np.max)
        
        num_colors = 256
        colors = [(0, 'black')]
        for i in range(1, num_colors):
            frequency = i / num_colors * (sr / 2)
            hue = frequency / (sr / 2)
            colors.append((i / (num_colors - 1), plt.cm.hsv(hue)[:3]))
        
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log', cmap=cmap, ax=ax)
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Decibels')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode()

        # Save the uploaded file temporarily
        temp_filename = 'uploaded_audio.wav' if file.filename.endswith('.wav') else 'uploaded_audio.mp3'
        file.save(temp_filename)
        
        plot_url2,frequencies,amplitudes,colorss = process_audio(y)
        
        # Store frequency data for Dash
        frequency_data = {
            'frequencies': frequencies,
            'amplitudes': amplitudes,
            'colors': colorss
        }
        #session['frequency_data'] = frequency_data

        df=pd.DataFrame(frequency_data)
        df.to_csv('freq.csv',index=0)
        

        return render_template('color_representation.html', plot_url=plot_url)

@app.route('/text-to-color', methods=['GET', 'POST'])
def text_to_color():
    if request.method == 'POST':
        input_string = ""
        if 'text' in request.form and request.form['text']:
            input_string = request.form['text']
        elif 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            filename = file.filename.lower()
            if filename.endswith('.txt'):
                input_string = file.read().decode('utf-8')
            elif filename.endswith('.docx'):
                input_string = read_docx(file)
            elif filename.endswith('.pdf'):
                input_string = read_pdf(file)
            else:
                return render_template('text_to_color.html', error="Unsupported file format")
        else:
            return render_template('text_to_color.html', error="No input provided")
        
        palette = generate_color_palette()
        image, color_code = string_to_color_pattern(input_string, palette)
        
        image_chunks = split_image_into_chunks(image, chunk_size=200)  

        return render_template('text_to_color_representation.html', image_chunks=image_chunks, color_code=color_code)

    return render_template('text_to_color.html')

@app.route('/color-to-text', methods=['GET', 'POST'])
def color_to_text():
    if request.method == 'POST':
        if 'color_code' in request.form and request.form['color_code']:
            color_code_input = request.form['color_code']
            print(color_code_input.strip())
            color_code = color_code_input.strip().strip('][').split('),')
            cz = []
            for x in color_code:
                z = x.strip().strip('(').strip(')')
                z = z.split(',')
                u,v,w = z
                z = (int(u),int(v),int(w))
                cz.append(tuple(z))
            print(cz)
            print(color_code)
            palette = generate_color_palette()
            original_text = color_code_to_string(cz, palette)
            return render_template('color_to_text.html', original_text=original_text, color_code=color_code_input)
        else:
            return render_template('color_to_text.html', error="No input provided")
    return render_template('color_to_text.html')

colors = plt.cm.Set1(np.linspace(0, 1, len(freqs_org)))  # Choose your preferred colormap
# Define frequency ranges and their corresponding colors
frequency_colors = {
    (100, 200): (255, 0, 0),          # Red
    (100, 300): (139, 0, 0),          # Dark Red
    (200, 300): (255, 127, 80),       # Coral
    (200, 400): (255, 165, 0),        # Orange
    (200, 600): (255, 215, 0),        # Gold
    (200, 1000): (255, 255, 0),       # Yellow
    (200, 1000): (255, 255, 224),     # Light Yellow
    (200, 1000): (255, 250, 205),     # Lemon
    (700, 1100): (0, 255, 0),         # Green
    (300, 1500): (0, 100, 0),         # Dark Green
    (1000, 2000): (50, 205, 50),      # Lime
    (1000, 3000): (128, 128, 0),      # Olive
    (400, 2000): (189, 252, 201),     # Mint
    (250, 2000): (0, 255, 255),       # Light Blue
    (250, 3000): (64, 224, 208),      # Turquoise
    (500, 2500): (46, 139, 87),       # Sea Wave
    (300, 3000): (135, 206, 235),     # Sky Blue
    (300, 3000): (0, 0, 255),         # Blue
    (300, 3000): (0, 0, 139),         # Dark Blue
    (500, 1500): (65, 105, 225),      # Royal Blue
    (500, 1500): (128, 0, 128),       # Violet
    (500, 2500): (221, 160, 221),     # Plum
    (500, 2500): (230, 230, 250),     # Lavender
    (1500, 4000): (255, 0, 255),      # Magenta
    (1700, 2000): (139, 0, 139),      # Dark Magenta
    (200, 5000): (255, 192, 203),     # Pink
    (2000, 5000): (255, 182, 193),    # Light Pink
    (2000, 8000): (250, 128, 114),    # Salmon
    (3000, 7000): (210, 105, 30),     # Chocolate
    (6000, 8000): (165, 42, 42),      # Brown
    (6000, 8000): (255, 140, 0)        # Dark Orange
}

frequency_colors = {
    (100, 200): (255, 0, 0),           # Red
    (100, 300): (139, 0, 0),           # Dark Red
    (200, 300): (255, 127, 80),        # Coral
    (200, 400): (255, 165, 0),         # Orange
    (200, 600): (255, 215, 0),         # Gold
    (200, 1000): (255, 255, 0),        # Yellow
    (200, 5000): (255, 192, 203),      # Pink
    (2000, 5000): (255, 182, 193),     # Light Pink
    (2000, 8000): (250, 128, 114),     # Salmon
    (300, 1500): (0, 100, 0),          # Dark Green
    (3000, 7000): (210, 105, 30),      # Chocolate
    (6000, 8000): (255, 140, 0)        # Dark Orange
    # Add remaining unique ranges and colors...
}

frequency_colors = {
    (100, 200): (255, 0, 0),           # /m/ as in "mat" (Red)
    (100, 200): (139, 0, 0),           # /p/ as in "pat" (Dark Red)
    (100, 300): (255, 127, 80),        # /b/ as in "bat" (Coral)
    (200, 400): (255, 165, 0),         # /d/ as in "dog" (Orange)
    (200, 600): (255, 215, 0),         # /g/ as in "go" (Gold)
    (200, 600): (255, 255, 0),         # /n/ as in "no" (Yellow)
    (200, 1000): (255, 255, 224),      # /w/ as in "wet" (Light Yellow)
    (200, 1000): (255, 250, 205),      # /r/ as in "rat" (Lemon)
    (200, 1000): (0, 255, 0),          # ‘ŋ’ (as in "sing") (Green)
    (700, 1100): (0, 100, 0),          # /ɑ/ as in "father" (Dark Green)
    (300, 1500): (50, 205, 50),        # /o/ as in "pot" (Lime)
    (1000, 2000): (128, 128, 0),       # /h/ as in "hat" (Olive)
    (1000, 3000): (189, 252, 201),     # ‘ð’ (as in "this") (Mint)
    (400, 2000): (0, 255, 255),        # /e/ as in "bed" (Light Blue)
    (250, 2000): (64, 224, 208),       # /u/ as in "put" (Turquoise)
    (250, 3000): (46, 139, 87),        # /i/ as in "sit" (Sea Wave)
    (500, 2500): (135, 206, 235),      # /a/ as in "cat" (Sky Blue)
    (300, 3000): (0, 0, 255),          # /l/ as in "lamp" (Blue)
    (300, 3000): (0, 0, 139),          # /t/ as in "top" (Dark Blue)
    (500, 1500): (65, 105, 225),       # ‘ʌ’ (as in "cup") (Royal Blue)
    (500, 1500): (128, 0, 128),        # ‘ə’ (as in "sofa") (Violet)
    (500, 2500): (221, 160, 221),      # /j/ as in "jump" (Plum)
    (500, 2500): (230, 230, 250),      # ‘æ’ (as in "cat") (Lavender)
    (1500, 4000): (255, 0, 255),       # /k/ as in "kite" (Magenta)
    (1700, 2000): (139, 0, 139),       # /f/ as in "fish" (Dark Magenta)
    (200, 5000): (255, 192, 203),      # /v/ as in "vet" (Pink)
    (2000, 5000): (255, 182, 193),     # /s/ as in "sat" (Light Pink)
    (2000, 5000): (250, 128, 114),     # ‘ʒ’ (as in "measure") (Salmon)
    (2000, 8000): (210, 105, 30),      # /ʃ/ as in "she" (Chocolate)
    (3000, 7000): (165, 42, 42),       # /z/ as in "zoo" (Brown)
    (6000, 8000): (255, 140, 0)        # ‘θ’ (as in "thin") (Dark Orange)
}

frequency_colors = {
    (27.50, 29.14): (139, 0, 0),        # A0 (Dark Red)
    (29.14, 30.87): (197, 34, 0),       # A#0/Bb0 (Dark Red)
    (30.87, 32.70): (255, 69, 0),       # B0 (Coral)
    (32.70, 34.65): (204, 204, 0),      # C1 (Yellow)
    (34.65, 36.71): (102, 152, 0),      # C#1/Db1 (Green)
    (36.71, 38.89): (0, 100, 0),        # D1 (Dark Green)
    (38.89, 41.20): (0, 50, 69),        # D#1/Eb1 (Blue)
    (41.20, 43.65): (41, 20, 0),        # E1 (Brown)
    (43.65, 46.25): (0, 0, 139),        # F1 (Dark Blue)
    (46.25, 49.00): (75, 0, 130),       # F#1/Gb1 (Purple)
    (49.00, 51.91): (112, 0, 171),      # G1 (Dark Purple)
    (51.91, 55.00): (148, 0, 211),      # G#1/Ab1 (Light Purple)
    (55.00, 58.27): (139, 0, 0),        # A1 (Dark Red)
    (58.27, 61.74): (197, 34, 0),       # A#1/Bb1 (Dark Red)
    (61.74, 65.41): (255, 69, 0),       # B1 (Coral)
    (65.41, 69.30): (204, 204, 0),      # C2 (Yellow)
    (69.30, 73.42): (102, 152, 0),      # C#2/Db2 (Green)
    (73.42, 77.78): (0, 100, 0),        # D2 (Dark Green)
    (77.78, 82.41): (0, 50, 69),        # D#2/Eb2 (Blue)
    (82.41, 87.31): (41, 20, 0),        # E2 (Brown)
    (87.31, 92.50): (0, 0, 139),        # F2 (Dark Blue)
    (92.50, 98.00): (75, 0, 130),       # F#2/Gb2 (Purple)
    (98.00, 103.83): (112, 0, 171),     # G2 (Dark Purple)
    (103.83, 110.00): (148, 0, 211),    # G#2/Ab2 (Light Purple)
    (110.00, 116.54): (139, 0, 0),      # A2 (Dark Red)
    (116.54, 123.47): (197, 34, 0),     # A#2/Bb2 (Dark Red)
    (123.47, 130.81): (255, 69, 0),     # B2 (Coral)
    (130.81, 138.59): (204, 204, 0),    # C3 (Yellow)
    (138.59, 146.83): (102, 152, 0),    # C#3/Db3 (Green)
    (146.83, 155.56): (0, 100, 0),      # D3 (Dark Green)
    (155.56, 164.81): (0, 50, 69),      # D#3/Eb3 (Blue)
    (164.81, 174.61): (41, 20, 0),      # E3 (Brown)
    (174.61, 185.00): (0, 0, 139),      # F3 (Dark Blue)
    (185.00, 196.00): (75, 0, 130),     # F#3/Gb3 (Purple)
    (196.00, 207.65): (112, 0, 171),    # G3 (Dark Purple)
    (207.65, 220.00): (148, 0, 211),    # G#3/Ab3 (Light Purple)
    (220.00, 233.08): (139, 0, 0),      # A3 (Dark Red)
    (233.08, 246.94): (197, 34, 0),     # A#3/Bb3 (Dark Red)
    (246.94, 261.63): (255, 69, 0),     # B3 (Coral)
    (261.63, 277.18): (204, 204, 0),    # C4 (Yellow)
    (277.18, 293.66): (102, 152, 0),    # C#4/Db4 (Green)
    (293.66, 311.13): (0, 100, 0),      # D4 (Dark Green)
    (311.13, 329.63): (0, 50, 69),      # D#4/Eb4 (Blue)
    (329.63, 349.23): (41, 20, 0),      # E4 (Brown)
    (349.23, 369.99): (0, 0, 139),      # F4 (Dark Blue)
    (369.99, 392.00): (75, 0, 130),     # F#4/Gb4 (Purple)
    (392.00, 415.30): (112, 0, 171),    # G4 (Dark Purple)
    (415.30, 440.00): (148, 0, 211),    # G#4/Ab4 (Light Purple)
    (440.00, 466.16): (139, 0, 0),      # A4 (Dark Red)
    (466.16, 493.88): (197, 34, 0),     # A#4/Bb4 (Dark Red)
    (493.88, 523.25): (255, 69, 0),     # B4 (Coral)
    (523.25, 554.37): (204, 204, 0),    # C5 (Yellow)
    (554.37, 587.33): (102, 152, 0),    # C#5/Db5 (Green)
    (587.33, 622.25): (0, 100, 0),      # D5 (Dark Green)
    (622.25, 659.25): (0, 50, 69),      # D#5/Eb5 (Blue)
    (659.25, 698.46): (41, 20, 0),      # E5 (Brown)
    (698.46, 739.99): (0, 0, 139),      # F5 (Dark Blue)
    (739.99, 783.99): (75, 0, 130),     # F#5/Gb5 (Purple)
    (783.99, 830.61): (112, 0, 171),    # G5 (Dark Purple)
    (830.61, 880.00): (148, 0, 211),    # G#5/Ab5 (Light Purple)
    (880.00, 932.33): (139, 0, 0),      # A5 (Dark Red)
    (932.33, 987.77): (197, 34, 0),     # A#5/Bb5 (Dark Red)
    (987.77, 1046.50): (255, 69, 0),    # B5 (Coral)
    (1046.50, 1108.73): (204, 204, 0),  # C6 (Yellow)
    (1108.73, 1174.66): (102, 152, 0),  # C#6/Db6 (Green)
    (1174.66, 1244.51): (0, 100, 0),    # D6 (Dark Green)
    (1244.51, 1318.51): (0, 50, 69),    # D#6/Eb6 (Blue)
    (1318.51, 1396.91): (41, 20, 0),    # E6 (Brown)
    (1396.91, 1479.98): (0, 0, 139),
    (1479.98, 1567.98): (75, 0, 130),   # F#6/Gb6 (Purple)
    (1567.98, 1661.22): (112, 0, 171),  # G6 (Dark Purple)
    (1661.22, 1760.00): (148, 0, 211),  # G#6/Ab6 (Light Purple)
    (1760.00, 1864.66): (139, 0, 0),    # A6 (Dark Red)
    (1864.66, 1975.53): (197, 34, 0),   # A#6/Bb6 (Dark Red)
    (1975.53, 2093.00): (255, 69, 0),   # B6 (Coral)
    (2093.00, 2217.46): (204, 204, 0),  # C7 (Yellow)
    (2217.46, 2349.32): (102, 152, 0),  # C#7/Db7 (Green)
    (2349.32, 2489.02): (0, 100, 0),    # D7 (Dark Green)
    (2489.02, 2637.02): (0, 50, 69),    # D#7/Eb7 (Blue)
    (2637.02, 2793.83): (41, 20, 0),    # E7 (Brown)
    (2793.83, 2959.96): (0, 0, 139),    # F7 (Dark Blue)
    (2959.96, 3135.96): (75, 0, 130),   # F#7/Gb7 (Purple)
    (3135.96, 3322.44): (112, 0, 171),  # G7 (Dark Purple)
    (3322.44, 3520.00): (148, 0, 211),  # G#7/Ab7 (Light Purple)
    (3520.00, 3729.31): (139, 0, 0),    # A7 (Dark Red)
    (3729.31, 3951.07): (197, 34, 0),   # A#7/Bb7 (Dark Red)
    (3951.07, 4186.01): (255, 69, 0)     # B7 (Coral)
}



# Extract frequencies and their colors
freqs = list(frequency_colors.keys())
colors = list(frequency_colors.values())



@app.route('/record', methods=['GET', 'POST'])
def record():
    if request.method == 'POST':
        if 'file' in request.files:
            # Handle file upload
            audio_file = request.files['file']
            if audio_file.filename.endswith(('.mp3', '.wav')):
                
                # Save the uploaded file temporarily
                temp_filename = 'uploaded_audio.wav' if audio_file.filename.endswith('.wav') else 'uploaded_audio.mp3'
                audio_file.save(temp_filename)

                # Read audio data using soundfile
                audio_data, sample_rate = sf.read(temp_filename)

                #audio_segment.export('uploaded_audio.wav', format='wav')

                #audio_data = np.array(audio_segment.get_array_of_samples())
                audio_data = audio_data.reshape(-1, 1)
                return process_audio(audio_data)
            else:
                return 'Unsupported file format. Please upload MP3 or WAV.', 400
        else:
            # Handle recording
            duration = int(request.form['duration'])
            print("Recording starts...")
            audio_data = sd.rec(int(duration * Fs), samplerate=Fs, channels=1, dtype='float64')
            sd.wait()
            print("Recording completed.")
            return process_audio(audio_data)

    return render_template('frequency_plot.html', spectrum_image=None)

def process_audio(audio_data):
    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
        print("Audio has more than one channel. Using the first channel.")
        audio_data = audio_data[:, 0]  # Select the first channel
    L = len(audio_data)
    Y = fft(audio_data.flatten())
    P2 = np.abs(Y / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]
    f = Fs * np.arange((L // 2) + 1) / L
    

    # Plot the frequency spectrum
    ff=[]
    rr=[]
    colorss=[]
    width=2
    plt.figure(figsize=(12, 6))
    for i, (freq_range, color) in enumerate(zip(freqs, colors)):
        idx = np.where((f >= freq_range[0]) & (f < freq_range[1]))
        
        
        if idx[0].size > 0 :
            
            

            num_samples = 20  # Number of samples to plot
            selected_indices = np.random.choice(idx[0], size=min(num_samples, idx[0].size), replace=False)

            #ff.extend(f[selected_indices])
            #rr.extend(P1[selected_indices])
            #colorss.extend([color]*len(f[selected_indices]))

            ff.extend(f[idx])
            rr.extend(P1[idx])
              
            '''
            ff.append(np.min(f[idx]))
            rr.append(np.mean(P1[idx]))

            ff.append(np.max(f[idx]))
            rr.append(np.mean(P1[idx]))

            ff.append(f[0])
            rr.append(np.max(P1[idx]))

            ff.append(f[-1])
            rr.append(np.max(P1[idx]))
            '''
            

            colorss.extend([color]*len(f[idx]))
            
            plt.bar([np.mean(f[idx]),np.mean(f[idx])], [np.mean(P1[idx]),np.max(P1[idx])], color=np.array(color) / 255.0,width=5)
            plt.bar([np.min(f[idx]),np.min(f[idx])], [np.mean(P1[idx]),np.max(P1[idx])], color=np.array(color) / 255.0,width=5)
            plt.bar([np.max(f[idx]),np.max(f[idx])], [np.mean(P1[idx]),np.max(P1[idx])], color=np.array(color) / 255.0,width=5)

            plt.bar([f[0],f[0]], [np.mean(P1[idx]),np.max(P1[idx])], color=np.array(color) / 255.0,width=5)
            plt.bar([f[-1],f[-1]], [np.mean(P1[idx]),np.max(P1[idx])], color=np.array(color) / 255.0,width=5)


            plt.bar(f[selected_indices], P1[selected_indices], color=np.array(color) / 255.0,width=5)

            
    plt.title('Frequency Spectrum', fontsize=12)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.ylim([-1e-3,np.max(P1)])
    plt.xlim([np.min(ff),min(1000,np.max(ff))])
    plt.grid(True)
    plt.legend(notes, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=12, fontsize='small')

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)

    # Encode the image to base64
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url,ff,rr,colorss
    #return f'<h2>Frequency Spectrum</h2><img src="data:image/png;base64,{plot_url}" alt="Frequency Spectrum">'

@app.route('/upload', methods=['POST'])
def upload_file():
    return record()  # Delegate to the record function

@dash_app.callback(
    Output('bar-chart', 'figure'),
    [Input('frequency-data', 'data')]
)

def update_bar_chart(frequency_data):
    if frequency_data is None:
        return go.Figure()  # Return empty figure if no data

    #frequency_data = session['frequency_data']
    #frequency_data = session.get('frequency_data', None)

    freqq=pd.read_csv('freq.csv')
                
    frequency_data=freqq
    frequencies = list(frequency_data['frequencies'])
    amplitudes = list(frequency_data['amplitudes'])
    colors = frequency_data['colors']
    #amplitudes = [a * 1e6 for a in amplitudes]  # Scale up by a million for better visibility


    # Convert color strings to RGBA format
    rgba_colors = [
        f'rgba({int(color[1:-1].split(",")[0])}, {int(color[1:-1].split(",")[1])}, {int(color[1:-1].split(",")[2])}, 0.6)'
        for color in colors
    ]

   
    
    #fig = go.Figure(data=[
    #    go.Bar(
    #        x=frequencies,
    #        y=amplitudes,
    #        marker_color=rgba_colors  )
    #])

    fig = go.Figure(data=[
        go.Bar(
            x=frequencies,
            y=amplitudes,
            marker_color='rgba(0, 100, 200, 0.6)'  # Example color
        )
    ])
    df = pd.DataFrame({'Frequency': frequencies, 'Amplitude': amplitudes})

    # Create bar chart
    fig = px.box(df, x='Frequency', y='Amplitude', points="all")

    # Update the box colors
    for i, box in enumerate(fig.data):
        #if i < len(rgba_colors):
        print(rgba_colors[i])
        box.marker.color = rgba_colors[i]  # Assign custom color to each box
      
    # Create a list to hold the box traces
    box_traces = []
    
    unique_frequencies = df['Frequency'].unique()

    min_frequencies = df['Frequency'].min()
    max_frequencies = df['Frequency'].max()

    step=60

    '''
    # Create a box trace for each unique frequency
    for i, freq in enumerate(unique_frequencies):
        freq_data = df[df['Frequency'] == freq]
        note_name = notes[min(int(freq // 2), len(notes) - 1)]  # Map frequency to the corresponding note
        
        box_traces.append(go.Box(
            y=freq_data['Amplitude'],
            name=note_name,  # Name the box with the frequency
            marker_color=rgba_colors[i % len(rgba_colors)]  # Cycle through colors
        ))
    '''    


    for i, note in enumerate(notes[:-1]):
        # Define the frequency range for the note
        lower_bound = freqs_org[i]   # 5% lower
        upper_bound = freqs_org[i+1]   # 5% upper

        #if i>=len(list(frequency_colors.keys())):
        #    continue

        #lower_bound=list(frequency_colors.keys())[i][0]
        #upper_bound=list(frequency_colors.keys())[i][1]
        

        
        
        # Filter amplitudes for frequencies within the defined range
        freq_data = df[(df['Frequency'] > lower_bound) & 
                                   (df['Frequency'] < upper_bound)]
        xx=freq_data.index
        
        #print(rgba_colors[i])
        if not freq_data.empty:
            box_traces.append(go.Box(
                y=freq_data['Amplitude'],  # Amplitudes on Y-axis
                name=note,  # Name the box with the note name
                marker_color=rgba_colors[xx[0]]  # Cycle through colors
            ))
        
    frange = np.arange(0,int(max(frequencies)),500)
    # Create the figure with all box traces
    fig = go.Figure(data=box_traces)
    fig.update_layout(title='Frequency vs Amplitude',
                      xaxis_title='Frequency',
                      yaxis_title='Amplitude')   # Set y-axis limits)


    #fig.update_layout(title='Frequency vs Amplitude', xaxis_title='Frequency (Hz)', yaxis_title='Amplitude')
    return fig


# Layout for Dash app
dash_app.layout = html.Div([
    dcc.Store(id='frequency-data', data={}),  # To store frequency data
    dcc.Graph(id='bar-chart',figure=update_bar_chart(None)),
    html.Div(id='graph-container', style={'display': 'none'})  # Hidden div for handling updates
])

# Callback to load frequency data from session when a request is made
@dash_app.callback(
    Output('frequency-data', 'data'),
    [Input('bar-chart', 'id')]  # Dummy input to trigger callback
)
def load_frequency_data(_):
    frequency_data = session.get('frequency_data', {})
    return frequency_data  # Return the data to the Store

if __name__ == '__main__':
     app.run(debug=True)

