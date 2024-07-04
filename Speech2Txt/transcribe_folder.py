import os
import sys
import argparse
import time
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from pydub import AudioSegment  # ffmpeg must be installed. For Windows: https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/
from moviepy.editor import VideoFileClip

# Load API key from .env file
env_path = os.path.join("..", '.env')  # Adjust the path as necessary
load_dotenv(env_path)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Function to convert mp4 into mp3 extracting the audio
def convert_mp4_to_mp3(file_path, intermediate_outputs_folder):
    new_file = os.path.join(intermediate_outputs_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}.mp3")
    if os.path.exists(new_file):
        print(f"{new_file} already exists. Skipping conversion.")
        return new_file
    
    try:
        audio = AudioSegment.from_file(file_path, "mp4")
        audio.export(new_file, format="mp3")
        return new_file
    except Exception as e:
        print(f"Error converting mp4 to mp3: {e}")
        sys.exit(1)

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
def transcribe_audio(file_path, intermediate_outputs_folder):
    segment_times = []
    transcription_times = []
    
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    
    if file_size > 25:
        audio_segments = segment_audio(file_path, segment_times, intermediate_outputs_folder)
    else:
        audio_segments = [file_path]
    
    transcripts = []
    for idx, segment in enumerate(audio_segments):
        segment_transcript_path = os.path.join(intermediate_outputs_folder, f"{os.path.splitext(os.path.basename(segment))[0]}_transcript.txt")
        if os.path.exists(segment_transcript_path):
            print(f"Transcript for segment {idx+1} already exists. Skipping transcription.")
            with open(segment_transcript_path, "r", encoding="utf-8") as file:
                transcript = file.read()
        else:
            start_time = time.time()
            print(f"Transcribing {segment}...")

            MAX_NUM_RETRIES = 3

            for _ in range(MAX_NUM_RETRIES):
                try:
                    transcript = call_transcription_api(segment)
                    break  # Break out of the loop if successful
                except Exception as e:
                    print(f"Error transcribing {segment}: {e}")
                    print("Retrying...")
            else:
                raise Exception("All retries failed. Unable to transcribe segment.")
            end_time = time.time()
            transcription_time = end_time - start_time
            transcription_times.append(transcription_time)

            # Save the transcript for each segment
            if intermediate_outputs_folder:
                save_transcript(segment_transcript_path, transcript)
            print(f"Transcript for segment {idx+1} saved to {segment_transcript_path}")
            print(f"Transcription time for segment {idx+1}: {transcription_time:.2f} seconds")
        
        transcripts.append(transcript)
    
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

def postprocess(transcript, output_folder, file_path):
    postprocessed_transcript_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_post_processed_transcript.txt")
    if os.path.exists(postprocessed_transcript_path):
        print(f"Postprocessed transcript already exists. Skipping postprocessing.")
        with open(postprocessed_transcript_path, "r", encoding="utf-8") as file:
            postprocessed_transcript = file.read()
    else:
        postprocessed_transcript = call_postprocess_api(transcript)
        save_transcript(postprocessed_transcript_path, postprocessed_transcript)
    return postprocessed_transcript

def segment_audio(file_path, segment_times, intermediate_outputs_folder, max_duration_ms=1200000):  # 20 minutes in milliseconds
    audio = AudioSegment.from_mp3(file_path)
    segments = []

    for i in range(0, len(audio), max_duration_ms):
        segment_file_path = os.path.join(intermediate_outputs_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_{i // max_duration_ms}.mp3")
        if os.path.exists(segment_file_path):
            print(f"Segment {i // max_duration_ms + 1} already exists. Skipping segmentation.")
            segments.append(segment_file_path)
            continue

        print(f"Processing segment {i // max_duration_ms + 1}... of {file_path}")
        start_time = time.time()  # Start timing for segment
        segment = audio[i:i + max_duration_ms]
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

def transcribe_file(file_path, output_folder, intermediate_outputs_folder=None):
    print(f"Transcribing {file_path}...")
    total_start_time = time.time()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if intermediate_outputs_folder and not os.path.exists(intermediate_outputs_folder):
        os.makedirs(intermediate_outputs_folder)
    
    # Check if mp4 file, convert to mp3 if necessary
    if file_path.endswith(".mp4"):
        print("Converting mp4 to mp3...")
        audio_conversion_start_time = time.time()
        file_path = convert_mp4_to_mp3(file_path, intermediate_outputs_folder or output_folder)
        audio_conversion_end_time = time.time()
        print(f"Conversion time: {audio_conversion_end_time - audio_conversion_start_time:.2f} seconds")
        
    # Segment and transcribe the audio file
    transcription_start_time = time.time()
    combined_transcript_path = os.path.join(intermediate_outputs_folder or output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_original_transcript.txt")
    if os.path.exists(combined_transcript_path):
        print(f"Combined transcript already exists. Skipping transcription.")
        with open(combined_transcript_path, "r", encoding="utf-8") as file:
            combined_transcript = file.read()
        segment_times = []  # Segmentation times are not needed as segmentation is skipped
        transcription_times = []  # Transcription times are not needed as transcription is skipped
    else:
        combined_transcript, segment_times, transcription_times = transcribe_audio(file_path, intermediate_outputs_folder or output_folder)
        transcription_end_time = time.time()
        if intermediate_outputs_folder:
            save_transcript(combined_transcript_path, combined_transcript)
        
    # Postprocess the combined transcript
    postprocess_start_time = time.time()
    print("Postprocessing transcript...")
    postprocessed_transcript = postprocess(combined_transcript, output_folder, file_path)
    postprocess_end_time = time.time()
    
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
    if transcription_times:
        print(f"Transcription time: {transcription_end_time - transcription_start_time:.2f} seconds")
        for i, transcription_time in enumerate(transcription_times):
            print(f"  Transcription {i+1} time: {transcription_time:.2f} seconds")
    else:
        print("Transcription not required")
    print(f"Postprocessing time: {postprocess_end_time - postprocess_start_time:.2f} seconds")

def process_folder(input_folder, output_folder, intermediate_outputs_folder=None):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp4"):
            file_path = os.path.join(input_folder, file_name)
            transcribe_file(file_path, output_folder, intermediate_outputs_folder)

'''
python transcribe_folder.py --input_folder <path_to_input_folder> --output_folder <path_to_output_folder> [--intermediate_outputs_folder <path_to_intermediate_outputs_folder>]
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe and process audio.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing MP4 files")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--intermediate_outputs_folder", type=str, required=False, help="Path to the intermediate outputs folder")

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    intermediate_outputs_folder = args.intermediate_outputs_folder

    if not os.path.isdir(input_folder):
        print(f"Input folder not found: {input_folder}")
        sys.exit(1)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    process_folder(input_folder, output_folder, intermediate_outputs_folder)
