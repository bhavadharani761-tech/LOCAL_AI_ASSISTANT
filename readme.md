
# Local AI Answer Assistant

A Windows desktop application that captures microphone audio, transcribes speech locally, and generates AI-powered answers using a local language model.

## Technologies Used
- Python
- PySide6
- sounddevice
- faster-whisper
- Ollama
- Llama 3.2 1B

## Features
- Start and Stop microphone recording
- Local speech-to-text transcription
- AI-generated answers
- Progressive answer generation
- Local LLM inference using Ollama

## Setup

Install the required libraries:

```bash
pip install -r requirements.txt
```

Install Ollama and download the model:

```bash
ollama pull llama3.2:1b
```

Run the application:

```bash
python main.py
```

Speech-to-text and LLM inference run locally.
