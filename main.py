from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

import torch
import os
import io

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# MODEL_PATH = "./model"
# processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
# model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)


# Load model from Hugging Face
HF_MODEL_ID = "jubaerkhan49/bangla_stt"
processor = Wav2Vec2Processor.from_pretrained(HF_MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(HF_MODEL_ID)

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Please upload a .wav file")

    # Read and limit file size (e.g., max 5MB)
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    # Reset file pointer using in-memory buffer
    file_stream = io.BytesIO(contents)

    def process_audio():
        # Load audio using soundfile (CPU friendly)
        waveform, sr = sf.read(file_stream)
        
        if sr != 16000:
            import librosa
            waveform = librosa.resample(waveform.T, orig_sr=sr, target_sr=16000).T
            sr = 16000

        # Make sure waveform is a tensor
        waveform_tensor = torch.tensor(waveform).float()
        if waveform_tensor.dim() > 1:
            waveform_tensor = waveform_tensor.mean(dim=1)  # convert stereo to mono

        input_values = processor(waveform_tensor, sampling_rate=sr, return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.decode(predicted_ids[0])

    transcription = await run_in_threadpool(process_audio)

    return {"transcription": transcription}
