
import sys
import json
import numpy as np
import sounddevice as sd
import requests
from PySide6.QtGui import QTextCursor
from faster_whisper import WhisperModel
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QMessageBox,
)


# ---------------- CONFIGURATION ----------------

SAMPLE_RATE = 16000
MODEL_NAME = "llama3.2:1b"
OLLAMA_URL = "http://localhost:11434/api/generate"


# ---------------- SPEECH TO TEXT ----------------

class TranscriptionWorker(QThread):

    finished_text = Signal(str)
    error = Signal(str)

    def __init__(self, audio):
        super().__init__()
        self.audio = audio

    def run(self):
        try:
            # Local Whisper model
            model = WhisperModel(
                "base",
                device="cpu",
                compute_type="int8"
            )

            segments, info = model.transcribe(
                self.audio,
                language="en",
                vad_filter=True
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
            )

            self.finished_text.emit(text)

        except Exception as e:
            self.error.emit(str(e))


# ---------------- LOCAL AI ANSWER ----------------

class AnswerWorker(QThread):

    answer_chunk = Signal(str)
    finished_answer = Signal()
    error = Signal(str)

    def __init__(self, question):
        super().__init__()
        self.question = question

    def run(self):

        try:
            prompt = (
                "You are a helpful AI assistant. "
                "Answer the following question clearly "
                "and naturally. Keep the answer concise.\n\n"
                f"Question: {self.question}\n\n"
                "Answer:"
            )

            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": True
                },
                stream=True,
                timeout=(10, 300)
            )

            response.raise_for_status()

            for line in response.iter_lines():

                if not line:
                    continue

                data = json.loads(line)

                if "response" in data:
                    self.answer_chunk.emit(
                        data["response"]
                    )

                if data.get("done", False):
                    break

            self.finished_answer.emit()

        except Exception as e:
            self.error.emit(str(e))


# ---------------- DESKTOP APPLICATION ----------------

class LocalAIAssistant(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Local AI Answer Assistant"
        )

        self.setMinimumSize(650, 600)

        self.audio_chunks = []
        self.stream = None
        self.recording = False

        self.transcription_worker = None
        self.answer_worker = None

        self.setup_ui()

    def setup_ui(self):

        layout = QVBoxLayout()

        title = QLabel(
            "Local AI Answer Assistant"
        )

        title.setStyleSheet(
            "font-size: 22px; font-weight: bold;"
        )

        layout.addWidget(title)

        description = QLabel(
            "Speech-to-text and AI answers run locally."
        )

        layout.addWidget(description)

        # Audio buttons

        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")

        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(
            self.start_recording
        )

        self.stop_button.clicked.connect(
            self.stop_recording
        )

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

        # Transcription section

        layout.addWidget(
            QLabel("Transcription")
        )

        self.transcription_box = QTextEdit()

        self.transcription_box.setPlaceholderText(
            "Your spoken question will appear here..."
        )

        layout.addWidget(
            self.transcription_box
        )

        # AI answer section

        self.suggest_button = QPushButton(
            "Suggest Answer"
        )

        self.suggest_button.clicked.connect(
            self.suggest_answer
        )

        layout.addWidget(
            self.suggest_button
        )

        layout.addWidget(
            QLabel("AI Answer")
        )

        self.answer_box = QTextEdit()

        self.answer_box.setReadOnly(True)

        self.answer_box.setPlaceholderText(
            "Your AI-generated answer will appear here..."
        )

        layout.addWidget(
            self.answer_box
        )

        self.status_label = QLabel(
            "Ready"
        )

        layout.addWidget(
            self.status_label
        )

        self.setLayout(layout)

    # ---------------- RECORD AUDIO ----------------

    def start_recording(self):

        self.audio_chunks = []

        try:

            self.stream = sd.InputStream(
             device=None,
            samplerate=16000,
            channels=1,
            dtype="float32",
            callback=self.audio_callback
            )
            self.stream.start()

            self.recording = True

            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            self.suggest_button.setEnabled(False)

            self.status_label.setText(
                "Recording... Speak now."
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "Microphone Error",
                str(e)
            )

    def audio_callback(
        self,
        indata,
        frames,
        time,
        status
    ):

        if self.recording:

            self.audio_chunks.append(
                indata.copy()
            )

    # ---------------- STOP AND TRANSCRIBE ----------------

    def stop_recording(self):

        self.recording = False

        if self.stream is not None:

            self.stream.stop()
            self.stream.close()

            self.stream = None

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.status_label.setText(
            "Transcribing locally..."
        )

        if not self.audio_chunks:

            self.status_label.setText(
                "No audio recorded."
            )

            self.suggest_button.setEnabled(True)

            return

        audio = np.concatenate(
            self.audio_chunks,
            axis=0
        ).flatten()

        self.transcription_worker = (
            TranscriptionWorker(audio)
        )

        self.transcription_worker.finished_text.connect(
            self.transcription_complete
        )

        self.transcription_worker.error.connect(
            self.show_error
        )

        self.transcription_worker.start()

    def transcription_complete(self, text):

        self.transcription_box.setPlainText(
            text
        )

        self.status_label.setText(
            "Transcription complete."
        )

        self.suggest_button.setEnabled(True)

    # ---------------- GENERATE ANSWER ----------------

    def suggest_answer(self):

        question = (
            self.transcription_box
            .toPlainText()
            .strip()
        )

        if not question:

            QMessageBox.warning(
                self,
                "Empty Question",
                "Please record a question first."
            )

            return

        self.answer_box.clear()

        self.suggest_button.setEnabled(False)

        self.status_label.setText(
            "Generating local AI answer..."
        )

        self.answer_worker = AnswerWorker(
            question
        )

        self.answer_worker.answer_chunk.connect(
            self.append_answer
        )

        self.answer_worker.finished_answer.connect(
            self.answer_complete
        )

        self.answer_worker.error.connect(
            self.show_error
        )

        self.answer_worker.start()

    def append_answer(self, chunk):

        cursor = self.answer_box.textCursor()

        cursor.movePosition(
        QTextCursor.MoveOperation.End
        )

        self.answer_box.setTextCursor(cursor)

        self.answer_box.insertPlainText(chunk)
    def answer_complete(self):

        self.suggest_button.setEnabled(True)

        self.status_label.setText(
            "Answer generated successfully."
        )

    def show_error(self, message):

        self.status_label.setText(
            "An error occurred."
        )

        self.suggest_button.setEnabled(True)

        QMessageBox.critical(
            self,
            "Error",
            message
        )

    def closeEvent(self, event):

        self.recording = False

        if self.stream is not None:

            self.stream.stop()
            self.stream.close()

        event.accept()


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = LocalAIAssistant()

    window.show()

    sys.exit(app.exec())
