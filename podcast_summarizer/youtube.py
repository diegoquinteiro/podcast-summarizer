import os
import click
import webvtt
import yt_dlp


def extract_from_youtube(
    podcast_url_or_path, transcription_file=None, language=None
):
    download_options = get_download_options()

    if not language:
        language = detect_language(podcast_url_or_path)

    download_options["subtitleslangs"] = [language]

    with yt_dlp.YoutubeDL(download_options) as ydl:
        click.echo("📺 Looking for YouTube subtitles...")
        ydl.download(podcast_url_or_path)

        subtitles_file = f"yt-audio.{language}.vtt"
        if os.path.isfile(subtitles_file):
            vtt = webvtt.read(subtitles_file)

            if not transcription_file:
                transcription_file = "yt-audio.txt"
            if not transcription_file.endswith(".txt"):
                transcription_file += ".txt"

            create_transcription_from_subtitles(transcription_file, vtt)

            podcast_url_or_path = transcription_file

        else:
            download_options["skip_download"] = False
            with yt_dlp.YoutubeDL(download_options) as ydl:
                click.echo(
                    "📺 Extracting audio from YouTube video \
                        (may take a while)..."
                )
                ydl.download(podcast_url_or_path)
                podcast_url_or_path = "yt-audio.m4a"
                click.echo("\r", nl=False)
                click.secho(
                    "💾 Saved YouTube audio to yt-audio.m4a", fg="green")

    return podcast_url_or_path


def create_transcription_from_subtitles(transcription_file, vtt):
    with open(transcription_file, "w") as f:
        transcript = ""
        lines = []
        for line in vtt:
            lines.extend(line.text.strip().splitlines())

        previous = None
        for line in lines:
            if line == previous:
                continue
            transcript += "\n" + line
            previous = line

        f.write(transcript)

    click.echo("\r", nl=False)
    click.secho(
        f"💾 Saved YouTube transcription to {transcription_file}", fg="green"
    )


def detect_language(podcast_url_or_path):

    download_options = get_download_options()

    with yt_dlp.YoutubeDL(download_options) as ydl:
        click.echo("🌎 Detecting original language...")
        ydl.download(podcast_url_or_path)

        info = ydl.extract_info(podcast_url_or_path, download=False)

        language = next(
            format["language"]
            for format in info["formats"]
            if format["format_note"] == "Default"
        )

        if language:
            click.secho(
                f"🗣️ Detected original language: {language}", fg="green")
        else:
            language = "en"
    return language


def get_download_options():
    return {
        "format": "bestaudio/best",
        "skip_download": True,
        "outtmpl": "yt-audio",
        "quiet": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
    }
