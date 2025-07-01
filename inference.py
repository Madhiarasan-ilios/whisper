import whisperx
import os

def transcribe_and_diarize(audio_file_path, hf_auth_token):
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return

    device = "cpu"
    batch_size = 16
    compute_type = "int8"

    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file_path)

    result = model.transcribe(audio, batch_size=batch_size)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_auth_token, device=device)

    diarize_segments = diarize_model(audio)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    def format_conversation(segments):
        formatted_output = []
        for segment in segments:
            speaker = segment.get("speaker", "Unknown Speaker")
            text = segment.get("text", "").strip()
            if speaker.startswith("SPEAKER_"):
                try:
                    speaker_id = int(speaker.split('_')[-1])
                    speaker_label = f"Speaker {speaker_id + 1}"
                except ValueError:
                    speaker_label = speaker
            else:
                speaker_label = speaker
            formatted_output.append(f"{speaker_label}: {text}")
        return "\n".join(formatted_output)

    final_output = format_conversation(result['segments'])
    print(final_output)


if __name__ == "__main__":
    audio_file_path = "sample.wav"
    hugging_face_token = ""

    transcribe_and_diarize(audio_file_path, hugging_face_token)