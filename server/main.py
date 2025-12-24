"""
헬스체크 서버
FastAPI를 사용하여 구현
"""
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM


# 전역 변수로 모델 저장
model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 및 종료 이벤트 처리"""
    global model, tokenizer
    
    # Startup
    port = int(os.getenv('PORT', '7860'))
    host = os.getenv('HOST', '0.0.0.0')
    
    # unbuffered 출력을 위해 sys.stdout.flush() 사용
    print(f"\n{'='*60}", flush=True)
    print("Health Check Server Started", flush=True)
    print(f"Version: 1.0.3", flush=True)
    print(f"Host: {host}", flush=True)
    print(f"Port: {port}", flush=True)
    print(f"Health Check: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/health", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # DistilGPT2 모델 로드
    print("Loading DistilGPT2 model...", flush=True)
    try:
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"✓ DistilGPT2 model loaded successfully", flush=True)
    except Exception as e:
        print(f"✗ Error loading model: {e}", flush=True)
        raise
    
    yield
    
    # Shutdown (필요시 추가)
    # 모델 메모리 해제
    model = None
    tokenizer = None
    print("Shutting down server...", flush=True)


app = FastAPI(lifespan=lifespan)

@app.get("/")
def greet_json():
    """루트 엔드포인트"""
    return {"Hello": "World!"}

@app.get("/health")
def health_check():
    """헬스체크 엔드포인트"""
    model_status = {
        "loaded": model is not None and tokenizer is not None,
        "name": "distilgpt2" if model is not None else None
    }
    
    # 모델이 로드되어 있으면 간단한 질문에 대한 응답 생성
    sample_question = "Hello"
    sample_response = None
    
    if model is not None and tokenizer is not None:
        try:
            # 간단한 텍스트 생성 (짧게)
            inputs = tokenizer.encode(sample_question, return_tensors="pt")
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + 20,  # 짧게 생성
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 원본 프롬프트를 제외한 생성된 부분만 추출
            sample_response = generated_text[len(sample_question):].strip()
        except Exception as e:
            sample_response = f"Error: {str(e)}"
    
    return {
        "status": "healthy",
        "service": "Health Check Server",
        "version": "1.0.3",
        "model": model_status,
        "sample": {
            "question": sample_question if model is not None else None,
            "response": sample_response
        }
    }

@app.post("/generate")
def generate_text(prompt: str, max_length: int = 50):
    """텍스트 생성 엔드포인트"""
    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}
    
    try:
        # 입력 텍스트를 토큰화
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # 텍스트 생성
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "max_length": max_length
        }
    except Exception as e:
        return {"error": str(e)}
