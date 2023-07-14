# podcast-summarizer

![Version](https://img.shields.io/pypi/v/podcast-summarizer)
![Python Version Support](https://img.shields.io/pypi/pyversions/podcast-summarizer)

Summarizes a podcast from an audio or transcription using ChatGPT.

## Instalation

```bash
pip install podcast-summarizer
```

## Usage

```bash
# Summarize a podcast audio from an URL.
podcast-summarizer http://.../audio.mp3

# Summarize a podcast audio from an YouTube video.
podcast-summarizer https://www.youtube.com/watch?v=...

# Summarize a podcast audio from local file.
podcast-summarizer audio.mp3

# Summarize a podcast audio and save transcription and summary to files.
podcast-summarizer -t transcription.txt -s summary.txt audio.mp3

# Summarize an already transcribed podcast
podcast-summarizer transcription.txt
```

## Setup

You'll need to install `ffmpeg` before running the script.

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg
```

The script expects an environment variable called `OPEN_API_KEY`.

1. Get an [OpenAI API Key](https://github.com/Significant-Gravitas/Auto-GPT#:~:text=Get%20an%20OpenAI-,API%20Key,-Download%20the%20latest)
2. Add the key to your `OPEN_API_KEY` environment variable:

- If you use `bash`:

    ```bash
    echo 'export OPENAI_API_KEY={your api key}' >> ~/.bash_profile && source ~/.bash_profile
    ```

- If yu use `zsh`:

    ```bash
    echo 'export OPENAI_API_KEY={your api key}' >> ~/.zshenv && source ~/.zshenv
    ```
