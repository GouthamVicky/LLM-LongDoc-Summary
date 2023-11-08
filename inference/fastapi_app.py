from fastapi import FastAPI,Response
from pydantic import BaseModel
from typing import List
from vllm import LLM, SamplingParams

from preprocess import generate_prompt
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import re


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

model = "GouthamVignesh/falcon-arxiv-long-summary-1B"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


@app.post("/generate/")
def generate_text(prompt_request: PromptRequest):
    prompt = prompt_request.prompt

    print("Generating Prompt")

    prompt = generate_prompt(prompt)

    print("Prompt Generated")
    print(prompt)
    #generated_text = llm.generate([prompt], sampling_params)[0].outputs[0].text
    results = " ".join([f"{seq['generated_text']}" for seq in pipeline(prompt, max_length=1024, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)])
    generated_text=results.split("### Summary :")[-1]
    generated_text = re.sub(r'\n+', '\n', generated_text)
    return {"prompt": prompt, "generated_text": generated_text}

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=False)
