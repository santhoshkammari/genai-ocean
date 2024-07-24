import asyncio
from threading import Thread
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from starlette.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

app = FastAPI()


class ChatRequest(BaseModel):
    messages: List[dict]
    max_new_tokens: int = 5000
    temperature: float = 1e-3
    top_p: float = 0.9


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct",
        torch_dtype=torch.float32,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    return model, tokenizer


model, tokenizer = load_model()
print(tokenizer.eos_token)


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
            text = text.replace(tokenizer.eos_token,"")
        yield text
        await asyncio.sleep(0.2)  # Small delay to allow for smooth streaming



@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(generate(request), media_type="text/event-stream")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)