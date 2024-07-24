import asyncio
from threading import Thread
from fastapi import FastAPI, HTTPException
from huggingface_hub import login
from pydantic import BaseModel
from typing import List, Optional
from starlette.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

app = FastAPI()
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class ChatRequest(BaseModel):
    messages: List[dict]
    max_new_tokens: int = 5000
    temperature: float = 1e-3
    top_p: float = 0.9


class ModelLoadRequest(BaseModel):
    model: str
    token: str
    torch_dtype: str = "float32"
    quantization: Optional[str] = None

@app.get("/model_status")
async def get_model_status():
    return {"loaded": model is not None and tokenizer is not None}

@app.post("/load_model")
async def load_model(request: ModelLoadRequest):
    global model, tokenizer
    try:
        if request.token:
            login(token=request.token)
        torch_dtype = getattr(torch, request.torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            request.model,
            torch_dtype=torch_dtype,
            device_map=device,
            load_in_8bit=(request.quantization == "8bit"),
            load_in_4bit=(request.quantization == "4bit")
        )
        tokenizer = AutoTokenizer.from_pretrained(request.model)
        return {"message": f"Model {request.model} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")


async def generate(request: ChatRequest):
    text = tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True)

    generate_kwargs = dict(
        input_ids=model_inputs.input_ids,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    for text in streamer:
        if tokenizer.eos_token in text:
            text = text.replace(tokenizer.eos_token, "")
        yield text
        await asyncio.sleep(0.2)  # Small delay to allow for smooth streaming


@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(generate(request), media_type="text/event-stream")


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
