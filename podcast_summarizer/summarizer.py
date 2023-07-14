import os
import openai
import urllib.request
import tempfile
import whisper
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

messages = []
SUPPORTED_AUDIO_FORMATS = ["m4a", "mp3", "webm", "mp4", "mpga", "wav", "mpeg"]


def openAudio(url_or_path, audio_format):
    if os.path.isfile(url_or_path):
        return open(url_or_path, "rb")
    try:
        if not audio_format:
            audio_format = os.path.splitext(url_or_path)[1]
            if not audio_format or audio_format not in SUPPORTED_AUDIO_FORMATS:
                audio_format = "mp3"

        audio_file = tempfile.NamedTemporaryFile(suffix=audio_format)
        with urllib.request.urlopen(url_or_path) as audio:
            audio_file.write(audio.read())
        return audio_file
    except ValueError:
        raise Exception("Invalid URL or path.")


def transcribe(mp3_path):
    model = whisper.load_model("tiny")
    result = model.transcribe(mp3_path, fp16=False)

    return result["text"]


def summarize(text, language):
    return askLLM(
        f'Please summarize the following podcast transcription, respond in "{language}": {text}'
    )


def askLLM(message):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logging.error("OPENAI_API_KEY not set", exc_info=True)

    global messages

    messages += [
        {
            "role": "user",
            "content": message,
        }
    ]

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=0.7,
        )
    except openai.error.RateLimitError as e:
        logging.error("RateLimitError occurred:", exc_info=True)
        logging.error("RateLimitError: %s", e)
        logging.error("Error details: %s", e.__dict__)

    response = completion.choices[0].message.content
    return response
