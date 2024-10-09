
import random
import textwrap
import argparse
import time
import numpy as np
import sounddevice as sd
import pyttsx3
from transformers import pipeline
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# Load the ASR model
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
# Initialize the TTS engine
tts_engine = pyttsx3.init()

# Define the model to be used
model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Replace with HF Actual Token and it is free, no cost!
# See hot to get one through this link below:
# [https://huggingface.co/docs/hub/en/security-tokens]
access_token = "REPLACE_WITH_TOKEN" 


client = InferenceClient(model=model, token=access_token)

# Embedded system prompt
system_prompt_text = (
    "You are a smart and helpful co-worker that answer normal questions briefly in around 100 words"
    "You help with any kind of request and provide 200 words if the topic is technical"
)


# Read the content of the info.md file
with open("info.md", "r", encoding="utf-8") as file:
    info_md_content = file.read()

# Chunk the info.md content into smaller sections
chunk_size = 2500  # Adjust this size as needed
info_md_chunks = textwrap.wrap(info_md_content, chunk_size)

def get_all_chunks(chunks):
    return "\n\n".join(chunks)

def format_prompt_mixtral(message, history, info_md_chunks):
    prompt = "<s>"
    all_chunks = get_all_chunks(info_md_chunks)
    prompt += f"{all_chunks}\n\n"  # Add all chunks of info.md at the beginning
    prompt += f"{system_prompt_text}\n\n"  # Add the system prompt

    if history:
        for user_prompt, bot_response in history:
            prompt += f"[INST] {user_prompt} [/INST] {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt

def chat_inf(prompt, history, seed, temp, tokens, top_p, rep_p):
    generate_kwargs = dict(
        temperature=temp,
        max_new_tokens=tokens,
        top_p=top_p,
        repetition_penalty=rep_p,
        do_sample=True,
        seed=seed,
    )

    formatted_prompt = format_prompt_mixtral(prompt, history, info_md_chunks)

    # Implement retry logic
    for attempt in range(5):
        try:
            stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
            output = ""
            for response in stream:
                output += response.token.text

            # Print the response for debugging
            print(f"Response: {output}")

            # Check if output is empty
            if not output:
                return history, "No response."

            history.append((prompt, output))
            return history, output
            
        except HfHubHTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)  # Wait before retrying
            else:
                print(f"HTTP Error: {e.response.status_code} - {e.message}")
                return history, "An error occurred during processing."

def record_audio(silence_threshold=0.001, silence_duration=1.5, sample_rate=16000):
    print("Listening... (Press Ctrl+C to stop)")
    audio_buffer = []
    silence_counter = 0
    max_silence_frames = int(silence_duration * sample_rate)
    
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
        while True:
            data = stream.read(1024)[0]
            audio_buffer.extend(data.flatten())  # Flatten the data to ensure it's mono

            # Calculate the volume level
            volume = np.abs(data).mean()
            
            print(f"Volume level: {volume:.2f}")

            if volume > silence_threshold:
                silence_counter = 0  # Reset the silence counter
            else:
                silence_counter += len(data)

            # Stop recording if silence is detected for the specified duration
            if silence_counter > max_silence_frames:
                print("Silence detected, stopping recording.")
                break

    return np.array(audio_buffer, dtype=np.float32), sample_rate  # Ensure it's a float32 array


def transcribe(audio, sr):
    # Normalize the audio data
    audio = audio.astype(np.float32)
    audio /= np.max(np.abs(audio)) if np.max(np.abs(audio)) != 0 else 1  # Avoid division by zero

    # Transcribe the audio
    return transcriber({"sampling_rate": sr, "raw": audio})["text"]

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def main():
    parser = argparse.ArgumentParser(description="PTT Chatbot CLI with Voice")
    parser.add_argument('--seed', type=int, default=random.randint(1, 1111111111111111), help='Random seed for generation')
    parser.add_argument('--temperature', type=float, default=0.9, help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=3840, help='Max new tokens')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
    parser.add_argument('--rep_penalty', type=float, default=1.0, help='Repetition penalty')

    args = parser.parse_args()

    history = []
    
    try:
        while True:
            # Record and transcribe the user's voice input
            audio, sr = record_audio()

            # Check if audio is recorded
            if audio.size == 0:
                transcription = "Please wait, I will be back. Just reply, 'I am waiting for you.' and if the message is cut off or incomplete just assume it was accident and reply the similarly"
            else:
                transcription = transcribe(audio, sr)
                print("Transcription:")
                print(transcription)

            # Generate response using the chatbot
            history, output = chat_inf(transcription, history, args.seed, args.temperature, args.max_tokens, args.top_p, args.rep_penalty)
            print(f"Bot: {output}")

            # Speak the response
            speak(output)
            print("\nListening again...\n")
    except KeyboardInterrupt:
        print("\nStopped listening.")

if __name__ == "__main__":
    main()
