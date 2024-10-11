import matplotlib
matplotlib.use('Agg') 
import colorsys
from flask import Flask, render_template, request
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

#import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from scipy.fft import fft
import io
import base64


app = Flask(__name__)

# Define the audio recording settings
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
freqs = [27.50, 29.14, 30.87, 32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.00, 51.91, 55.00, 58.27, 
         61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 
         130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 
         261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 
         523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 
         1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.00, 
         1864.66, 1975.53, 2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 
         3322.44, 3520.00, 3729.31, 3951.07, 4186.01]

# Set colors for each frequency
colors = [[139/255, 0, 0]] * len(freqs)

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
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    if file:
        y, sr = librosa.load(file)
        
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
        plot_url2 = process_audio(y)

        return render_template('color_representation.html', plot_url=plot_url,plot_url2=plot_url2)

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

colors = plt.cm.Set1(np.linspace(0, 1, len(freqs)))  # Choose your preferred colormap
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

# Extract frequencies and their colors
freqs = list(frequency_colors.keys())
colors = list(frequency_colors.values())

import soundfile as sf

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
    width=2
    plt.figure(figsize=(12, 6))
    for i, (freq_range, color) in enumerate(zip(freqs, colors)):
        idx = np.where((f >= freq_range[0]) & (f < freq_range[1]))
        if idx[0].size > 0 :
            ff.extend(f[idx])

            num_samples = 20000  # Number of samples to plot
            selected_indices = np.random.choice(idx[0], size=min(num_samples, idx[0].size), replace=False)
                    
            plt.scatter(f[selected_indices], P1[selected_indices], color=np.array(color) / 255.0)

    plt.title('Frequency Spectrum', fontsize=12)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.ylim([0,.7*np.max(P1)])
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
    return plot_url
    #return f'<h2>Frequency Spectrum</h2><img src="data:image/png;base64,{plot_url}" alt="Frequency Spectrum">'


@app.route('/upload', methods=['POST'])
def upload_file():
    return record()  # Delegate to the record function

if __name__ == '__main__':
    app.run(debug=True)

