🍼 NapNav: Smart Baby Monitoring System
NapNav is an AI-powered baby monitoring system that detects baby states, analyzes sleep patterns, predicts next nap times, plays soothing music, and gives parents important reminders — all via a web dashboard and installable PWA.

📌 Features
🖼️ Baby State Detection

Classifies baby state as awake, sleep, or cry using a PyTorch Faster R-CNN model.

📊 Sleep Pattern Analysis

Logs state changes into sleep_log.csv and visualizes historical trends.

🤖 Sleep Prediction

Predicts when the baby is likely to sleep next using a Random Forest Regressor.

🎵 Soothing Music Playback

Automatically plays soothing_music.mp3 when the baby is awake or crying.

💬 Built-in Chatbot

Chat interface for parenting queries.

📱 Progressive Web App

Works offline and is installable on mobile/desktop.

📂 Project Structure
graphql
Copy
Edit
NapNav/
│
├── app.py                       # Main Flask server
├── baby_sleep_analyzer.py       # Analyzes sleep patterns and predicts naps
├── baby_state_model.pth         # Trained PyTorch baby state model
├── calculate_accuracy.py        # Script to calculate model accuracy
├── calculate_prediction_accuracy.py  # Accuracy for prediction model
├── camera_monitor.py            # Monitors camera feed for real-time detection
├── chatbot.py                   # AI chatbot logic
├── dataset.py                   # Dataset handling for training/testing
├── image_to_timestamp_logger.py # Watches folder & logs baby states
├── model.py                     # Model architecture definition
├── monitor_and_play_music.py    # Plays soothing music when required
├── Model_Demonstration.ipynb    # Jupyter notebook demo
├── requirements.txt             # Dependencies
├── service-worker.js            # PWA offline caching
├── sleep_log.csv                # Historical baby states
├── soothing_music.mp3           # Audio for baby soothing
├── test_model.py                 # Script to test trained model
├── train.py                     # Model training script
│
├── data/                        # Dataset storage
├── new_images_to_log/           # Folder watched for new baby images
├── static/                      # Static assets (CSS, JS, icons)
├── templates/                   # HTML templates
│   ├── baby_details.html
│   ├── dashboard.html
│   └── login.html
└── venv/                        # Virtual environment
🔄 Workflow
Image Capture

Images are saved into new_images_to_log/.

State Detection

image_to_timestamp_logger.py uses baby_state_model.pth to classify the image.

Data Logging

Logs state & timestamp into sleep_log.csv.

Sleep Prediction

baby_sleep_analyzer.py reads log data and predicts the next nap time.

Music Playback

If baby is awake/crying → monitor_and_play_music.py plays soothing_music.mp3.

Dashboard Display

Flask (app.py) serves dashboard.html with live notifications, graphs, chatbot, and reminders.

🚀 Installation & Setup
1️⃣ Clone Repository

bash
Copy
Edit
git clone https://github.com/Ramyashree-20/Baby-Monitoring-System.git
cd Baby-Monitoring-System
2️⃣ Create Virtual Environment

bash
Copy
Edit
python -m venv venv
3️⃣ Activate Environment

bash
Copy
Edit
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
4️⃣ Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
5️⃣ Run Application

bash
Copy
Edit
python app.py
6️⃣ Open Browser

cpp
Copy
Edit
http://127.0.0.1:5000
🧠 Machine Learning Models
Baby State Detection

Model: Faster R-CNN (PyTorch)

Classes: awake, sleep, cry

Trained using dataset from data/

Sleep Prediction

Model: Random Forest Regressor

Input: Historical sleep/awake durations from sleep_log.csv

Output: Predicted next sleep time

