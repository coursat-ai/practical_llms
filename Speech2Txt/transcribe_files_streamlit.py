import os
import time
import shutil
import streamlit as st
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
        st.write(f"{new_file} already exists. Skipping conversion.")
        return new_file
    
    try:
        audio = AudioSegment.from_file(file_path, "mp4")
        audio.export(new_file, format="mp3")
        return new_file
    except Exception as e:
        st.write(f"Error converting mp4 to mp3: {e}")
        return None

# Function to call transcription API using OpenAI Whisper
def call_transcription_api(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            timestamp_granularities=["segment"]
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
        segment_transcript_path = os.path.join(intermediate_outputs_folder, f"{os.path.splitext(os.path.basename(segment))[0]}_original_transcript.txt")
        if os.path.exists(segment_transcript_path):
            st.write(f"Transcript for segment {idx+1} already exists. Skipping transcription.")
            with open(segment_transcript_path, "r", encoding="utf-8") as file:
                transcript = file.read()
        else:
            start_time = time.time()
            st.write(f"Transcribing {segment}...")

            MAX_NUM_RETRIES = 3

            for _ in range(MAX_NUM_RETRIES):
                try:
                    transcript = call_transcription_api(segment)
                    break  # Break out of the loop if successful
                except Exception as e:
                    st.write(f"Error transcribing {segment}: {e}")
                    st.write("Retrying...")
            else:
                raise Exception("All retries failed. Unable to transcribe segment.")
            end_time = time.time()
            transcription_time = end_time - start_time
            transcription_times.append(transcription_time)

            # Save the transcript for each segment
            if intermediate_outputs_folder:
                save_transcript(segment_transcript_path, transcript)
                st.write(f"Transcript for segment {idx+1} saved to {segment_transcript_path}")
            st.write(f"Transcription time for segment {idx+1}: {transcription_time:.2f} seconds")
        
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
                    Provide your output in English."""
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
        st.write(f"Postprocessed transcript already exists. Skipping postprocessing.")
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
            st.write(f"Segment {i // max_duration_ms + 1} already exists. Skipping segmentation.")
            segments.append(segment_file_path)
            continue

        st.write(f"Processing segment {i // max_duration_ms + 1}... of {file_path}")
        start_time = time.time()  # Start timing for segment
        segment = audio[i:i + max_duration_ms]
        segment.export(segment_file_path, format="mp3")
        segments.append(segment_file_path)
        end_time = time.time()  # End timing for segment
        
        segment_time = end_time - start_time
        segment_times.append(segment_time)
        st.write(f"Segment {i // max_duration_ms + 1} time: {segment_time:.2f} seconds")  # Log segment time
    
    return segments

def save_transcript(file_name, content):
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(content)

def transcribe_file(file_path, output_folder, intermediate_outputs_folder=None):
    st.write(f"Transcribing {file_path}...")
    total_start_time = time.time()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if intermediate_outputs_folder and not os.path.exists(intermediate_outputs_folder):
        os.makedirs(intermediate_outputs_folder)
    
    # Check if mp4 file, convert to mp3 if necessary
    if file_path.endswith(".mp4"):
        st.write("Converting mp4 to mp3...")
        audio_conversion_start_time = time.time()
        file_path = convert_mp4_to_mp3(file_path, intermediate_outputs_folder or output_folder)
        audio_conversion_end_time = time.time()
        st.write(f"Conversion time: {audio_conversion_end_time - audio_conversion_start_time:.2f} seconds")
        
    # Segment and transcribe the audio file
    transcription_start_time = time.time()
    combined_transcript_path = os.path.join(intermediate_outputs_folder or output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_original_transcript.txt")
    if os.path.exists(combined_transcript_path):
        st.write(f"Combined transcript already exists. Skipping transcription.")
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
    st.write("Postprocessing transcript...")
    postprocessed_transcript = postprocess(combined_transcript, output_folder, file_path)
    postprocess_end_time = time.time()
    
    total_end_time = time.time()
    
    # Print timing information
    st.write("\nTiming Information:")
    st.write(f"Total time: {total_end_time - total_start_time:.2f} seconds")
    if segment_times:
        st.write(f"Segmentation time: {sum(segment_times):.2f} seconds")
        for i, segment_time in enumerate(segment_times):
            st.write(f"  Segment {i+1} time: {segment_time:.2f} seconds")
    else:
        st.write("Segmentation not required")
    if transcription_times:
        st.write(f"Transcription time: {transcription_end_time - transcription_start_time:.2f} seconds")
        for i, transcription_time in enumerate(transcription_times):
            st.write(f"  Transcription {i+1} time: {transcription_time:.2f} seconds")
    else:
        st.write("Transcription not required")
    st.write(f"Postprocessing time: {postprocess_end_time - postprocess_start_time:.2f} seconds")

# Streamlit UI
st.title("ideo Transcriber")

uploaded_files = st.file_uploader("Choose MP4 files", accept_multiple_files=True, type=["mp4"])
output_folder = st.text_input("Enter the output folder path:")
intermediate_outputs_folder = st.text_input("Enter the intermediate outputs folder path:")
keep_intermediate_outputs = st.checkbox("Keep intermediate outputs", value=True)

if st.button("Transcribe"):
    if uploaded_files and output_folder and intermediate_outputs_folder:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists(intermediate_outputs_folder):
            os.makedirs(intermediate_outputs_folder)
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(intermediate_outputs_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.write(f"Processing file: {file_path}")

            with st.spinner(f"Transcribing {uploaded_file.name}..."):
                transcribe_file(file_path, output_folder, intermediate_outputs_folder)

        if not keep_intermediate_outputs:
            try:
                shutil.rmtree(intermediate_outputs_folder)
                st.write(f"Deleted intermediate outputs folder: {intermediate_outputs_folder}")
            except PermissionError as e:
                st.write(f"PermissionError: {e}. Trying alternative deletion method.")
                try:
                    # Alternative deletion method
                    for root, dirs, files in os.walk(intermediate_outputs_folder, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    os.rmdir(intermediate_outputs_folder)
                    print(f"Successfully deleted intermediate outputs folder using alternative method: {intermediate_outputs_folder}")
                except Exception as e:
                    print(f"Error deleting intermediate outputs folder: {e}")
    else:
        st.write("Please select files and enter all required folder paths.")
