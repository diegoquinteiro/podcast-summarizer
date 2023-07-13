import os
import click
import click_spinner
import podcast_summarizer.youtube as youtube
import podcast_summarizer.summarizer as summarizer
from shutil import which

@click.command()
@click.version_option()
@click.option('--summary-file', '-s', default=None, help="The file to save the summary to.")
@click.option('--transcription-file', '-t', default=None, help="The file to save the transcription to.")
@click.option('--audio-format', '-f', default=None, help="The format of the audio file. Defaults to mp3 when it cannot be infered.")
@click.option('--language', '-l', default="en", help="The language code for the language of the podcast.")
@click.argument('podcast_url_or_path')
def cli(podcast_url_or_path, summary_file, transcription_file, audio_format, language):
    """Summarizes the podcast provided by the PODCAST_URL_OR_PATH.
    
    - PODCAST_URL_OR_PATH can be a URL or a path to a local audio file.
    
    - PODCAST_URL_OR_PATH can be a YouTube URL (https://www.youtube.com/watch?v=...)

    - PODCAST_URL_OR_PATH can also be a path to a local ".txt" file containing the transcription.
    """

    if not which("ffmpeg"):
        raise Exception("ffmpeg is not installed. Please install ffmpeg.")

    if not os.getenv("OPENAI_API_KEY"):
        raise Exception("OPENAI_API_KEY is not set. Please add your API key to your environment variables.")
    
    if podcast_url_or_path.startswith("https://www.youtube.com/watch?v="):
        podcast_url_or_path = youtube.extract_from_youtube(podcast_url_or_path, transcription_file, language)
    
    if podcast_url_or_path.lower().endswith(".txt") and os.path.isfile(podcast_url_or_path):
        transcription = open(podcast_url_or_path, "r").read()
    else:
        with summarizer.openAudio(podcast_url_or_path, audio_format) as file:
            click.echo("🎙️ Transcribing...", nl=False)
            with click_spinner.spinner():
                transcription = summarizer.transcribe(file.name)
            click.echo("\r", nl=False)
            click.secho("🎙️ Transcription complete!", fg="green")

        if transcription_file:
            with open(transcription_file, "w") as transcription_file:
                transcription_file.write(transcription)
                click.secho("💾 Transcription saved to transcription.txt")
        
    click.echo("📝 Summarizing...", nl=False)
    with click_spinner.spinner():
        summary = summarizer.summarize(transcription, language)
    click.echo("\r", nl=False)
    click.secho("📝 Summary complete!", fg="green")

    if summary_file:
        with open(summary_file, "w") as summary_file:
            summary_file.write(summary)
            click.secho("💾 Summary saved to summary.txt", fg="green")

    click.echo("\n", nl=False)
    click.echo(summary)

    click.secho("\nDo you have any questions about the podcast? (Or type 'exit' to quit)", fg="blue")

    prompt = click.prompt(click.style("\n> ", fg="bright_white"), prompt_suffix="")
    while prompt.lower() != "exit":
        click.echo(summarizer.askLLM(prompt))
        prompt = click.prompt(click.style("\n> ", fg="bright_white"), prompt_suffix="")


if __name__ == '__main__':
    cli()