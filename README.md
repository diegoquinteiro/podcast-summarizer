# podcast-summarizer
![Version](https://img.shields.io/pypi/v/podcast-summarizer)
![Python Version Support](https://img.shields.io/pypi/pyversions/podcast-summarizer)

Summarizes a podcast from an audio or transcription using ChatGPT.

## Usage
```bash
# Summarize a podcast audio from an URL.
podcast-summarizer http://.../audio.mp3

# Summarize a podcast audio from local file.
podcast-summarizer audio.mp3

# Summarize a podcast audio and save transcription and summary to files.
podcast-summarizer -t transcription.txt -s summary.txt http://.../audio.mp3

# Summarize an already transcribed podcast
podcast-summarizer transcription.txt
```

## Instalation
```bash
pip install podcast-summarizer
```

You'll need to install `ffmpeg` before running the script.

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg
```

You'll also need to set the `OPENAI_API_KEY` environment variable.

```bash
export OPENAI_API_KEY=<YOUR_API_KEY..>
```