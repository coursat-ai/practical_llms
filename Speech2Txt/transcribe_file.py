import os
import sys
import argparse
import time
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from pydub import AudioSegment

# Load API key from .env file
env_path = os.path.join("..", '.env')  # Adjust the path as necessary
load_dotenv(env_path)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Function to convert mp4 into mp3 extracting the audio
def convert_mp4_to_mp3(file_path):
    base, ext = os.path.splitext(file_path)
    new_file = base + ".mp3"
    os.system(f"ffmpeg -i {file_path} -vn -acodec libmp3lame -y {new_file}")
    return new_file

# Function to call transcription API using OpenAI Whisper
def call_transcription_api(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcript

# Function to segment and transcribe audio
def transcribe_audio(file_path):
    segment_times = []
    transcription_times = []
    
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    
    if file_size > 25:
        audio_segments = segment_audio(file_path, segment_times)
    else:
        audio_segments = [file_path]
    
    transcripts = []
    for idx, segment in enumerate(audio_segments):
        start_time = time.time()
        print(f"Transcribing {segment}...")
        transcript = call_transcription_api(segment)
        end_time = time.time()
        transcription_time = end_time - start_time
        transcription_times.append(transcription_time)
        transcripts.append(transcript)

        # Save the transcript for each segment
        segment_transcript_path = f"{os.path.splitext(segment)[0]}_transcript.txt"
        save_transcript(segment_transcript_path, transcript)
        print(f"Transcript for segment {idx+1} saved to {segment_transcript_path}")
        print(f"Transcription time for segment {idx+1}: {transcription_time:.2f} seconds")
    
    combined_transcript = "\n".join(transcripts)
    return combined_transcript, segment_times, transcription_times
def call_postprocess_api(transcript):
    # Use GPT-4 to postprocess the transcript
    # Transcribe in the same language as the video
    system_prompt = """The following is a transcript of a video in Arabic. 
                    Please improve it and make it more readable. 
                    Do not summarize the content.
                    If a term is mentioned in English, keep it as is.
                    Provide your output in the same language of the video."""
    postprocessed_transcript = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": transcript,
        }
    ],
    temperature=0,
    #model="gpt-4-1106-preview",
    model="gpt-4o",
    )
    return postprocessed_transcript.choices[0].message.content

def postprocess(transcript):
    postprocessed_transcript = call_postprocess_api(transcript)
    return postprocessed_transcript


def segment_audio(file_path, segment_times, max_duration_ms=600000):  # 10 minutes in milliseconds
    audio = AudioSegment.from_mp3(file_path)
    segments = []

    for i in range(0, len(audio), max_duration_ms):
        start_time = time.time()  # Start timing for segment
        segment = audio[i:i + max_duration_ms]
        segment_file_path = f"{os.path.splitext(file_path)[0]}_{i // max_duration_ms}.mp3"
        segment.export(segment_file_path, format="mp3")
        segments.append(segment_file_path)
        end_time = time.time()  # End timing for segment
        
        segment_time = end_time - start_time
        segment_times.append(segment_time)
        print(f"Segment {i // max_duration_ms + 1} time: {segment_time:.2f} seconds")  # Log segment time
    
    return segments

def save_transcript(file_name, content):
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(content)

def main(file_path):
    total_start_time = time.time()
    
    # Check if mp4 file, convert to mp3 if necessary
    if file_path.endswith(".mp4"):
        print("Converting mp4 to mp3...")
        audio_conversion_start_time = time.time()
        file_path = convert_mp4_to_mp3(file_path)
        audio_conversion_end_time = time.time()
        print(f"Conversion time: {audio_conversion_end_time - audio_conversion_start_time:.2f} seconds")
        
    # Segment and transcribe the audio file
    transcription_start_time = time.time()
    combined_transcript, segment_times, transcription_times = transcribe_audio(file_path)
    transcription_end_time = time.time()
    save_transcript(f"{os.path.splitext(file_path)[0]}_original_transcript.txt", combined_transcript)
    
    # Postprocess the combined transcript
    postprocess_start_time = time.time()
    print("Postprocessing transcript...")
    postprocessed_transcript = postprocess(combined_transcript)
    postprocess_end_time = time.time()
    save_transcript(f"{os.path.splitext(file_path)[0]}_post_processed_transcript.txt", postprocessed_transcript)
    
    total_end_time = time.time()
    
    # Print timing information
    print("\nTiming Information:")
    print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
    if segment_times:
        print(f"Segmentation time: {sum(segment_times):.2f} seconds")
        for i, segment_time in enumerate(segment_times):
            print(f"  Segment {i+1} time: {segment_time:.2f} seconds")
    else:
        print("Segmentation not required")
    print(f"Transcription time: {transcription_end_time - transcription_start_time:.2f} seconds")
    for i, transcription_time in enumerate(transcription_times):
        print(f"  Transcription {i+1} time: {transcription_time:.2f} seconds")
    print(f"Postprocessing time: {postprocess_end_time - postprocess_start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe and process audio.")
    parser.add_argument("file_path", type=str, help="Path to the audio file")

    args = parser.parse_args()
    file_path = args.file_path

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    main(file_path)
