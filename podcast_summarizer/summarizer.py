import os
import openai
import urllib.request
import tempfile
import tiktoken
import whisper
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

messages = []
SUPPORTED_AUDIO_FORMATS = ["m4a", "mp3", "webm", "mp4", "mpga", "wav", "mpeg"]
TOKEN_LIMIT = 8192


def openAudio(url_or_path, audio_format):
    '''Opens an audio file from a URL or path.'''

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


def summarize(text, language=None):
    '''Summarizes the podcast using the LLM model.'''

    if num_tokens_from_string(text) > TOKEN_LIMIT:
        # Split text into multiple parts and summarize each part
        text = compress_text(text, language)

    if language:
        return ask_LLM(
            f"Please summarize the following podcast transcription,\
                respond in \"{language}\": {text}"
        )
    else:
        return ask_LLM(
            f'Please summarize the following podcast transcription,\
                respond in the same language as the transcription: {text}'
        )


def compress_text(text, language=None):
    '''Compresses the text by recursively summarizing it in two parts.'''

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)

    # Split token list in two
    first_tokens = tokens[: len(tokens) // 2]
    second_tokens = tokens[len(tokens) // 2:]

    if len(first_tokens) > TOKEN_LIMIT:
        first_tokens = encoding.encode(
            compress_text(encoding.decode(first_tokens)))

    if len(second_tokens) > TOKEN_LIMIT:
        second_tokens = encoding.encode(
            compress_text(encoding.decode(second_tokens)))

    # Summarize each half
    first_half = compress_through_LLM(encoding.decode(first_tokens), language)
    second_half = compress_through_LLM(
        encoding.decode(second_tokens), language)

    # Combine summaries
    return first_half + '' + second_half


def compress_through_LLM(text, language=None):
    '''Summarizes the text using the LLM model.'''
    if language:
        return ask_LLM(
            f"Please summarize the following podcast transcription,\
                respond in \"{language}\": {text}", False)
    else:
        return ask_LLM(
            f"Please make a summarized version of this podcast transcription:\
              {text}", False)


def ask_LLM(message, chat=True):
    if chat:
        global messages
    else:
        messages = []

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logging.error("OPENAI_API_KEY not set", exc_info=True)

    if num_tokens_from_string(message) > TOKEN_LIMIT:
        logging.error("Message exceeds token limit", exc_info=True)

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


def num_tokens_from_string(string, encoding_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
