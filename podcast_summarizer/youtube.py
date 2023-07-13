import os
import click
import webvtt
import yt_dlp


def extract_from_youtube(podcast_url_or_path, transcription_file=None, language="en"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'skip_download': True,
        'outtmpl': 'yt-audio',
        'quiet': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': [language],
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        click.echo("📺 Looking for YouTube subtitles...")
        ydl.download(podcast_url_or_path)
        if os.path.isfile(f"yt-audio.{language}.vtt"):
            vtt = webvtt.read(f"yt-audio.{language}.vtt")
            
            if not transcription_file:
                transcription_file = "yt-audio.txt"
            if not transcription_file.endswith(".txt"):
                transcription_file += ".txt"

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
            
            podcast_url_or_path = transcription_file
            click.echo("\r", nl=False)
            click.secho(f"💾 Saved YouTube transcription to {transcription_file}", fg="green")
            
        else:
            ydl_opts['skip_download'] = False
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                click.echo("📺 Extracting audio from YouTube video (may take a while)...")
                ydl.download(podcast_url_or_path)
                podcast_url_or_path = "yt-audio.m4a"
                click.echo("\r", nl=False)
                click.secho("💾 Saved YouTube audio to yt-audio.m4a", fg="green")
    
    return podcast_url_or_path