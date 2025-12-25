"""
헬스체크 서버
FastAPI를 사용하여 구현
llama-cpp-python을 사용한 LLaMA 모델 서빙
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

try:
    from llama_cpp import Llama
except ImportError:
    print("Warning: llama-cpp-python is not installed. Install it with: pip install llama-cpp-python", flush=True)
    Llama = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Warning: huggingface-hub is not installed. Install it with: pip install huggingface-hub", flush=True)
    hf_hub_download = None


# 전역 변수로 모델 저장
llama_model = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 및 종료 이벤트 처리"""
    global llama_model
    
    # Startup
    port = int(os.getenv('PORT', '7860'))
    host = os.getenv('HOST', '0.0.0.0')
    
    # unbuffered 출력을 위해 sys.stdout.flush() 사용
    print(f"\n{'='*60}", flush=True)
    print("LLaMA.cpp Server Starting...", flush=True)
    print(f"Version: 2.1.0", flush=True)
    print(f"Host: {host}", flush=True)
    print(f"Port: {port}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 모델 파일 경로 확인 및 다운로드
    model_path = os.getenv('MODEL_PATH', None)
    hf_model_id = os.getenv('HF_MODEL_ID', None)
    hf_filename = os.getenv('HF_FILENAME', None)  # 선택적, GGUF 파일명 지정
    
    # 환경변수가 없으면 기본 테스트 모델 사용
    if not model_path and not hf_model_id:
        print("ℹ No MODEL_PATH or HF_MODEL_ID specified, using default test model", flush=True)
        hf_model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
        hf_filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        print(f"  Default model: {hf_model_id}/{hf_filename}", flush=True)
    
    # Hugging Face Hub에서 모델 다운로드
    if hf_model_id:
        if hf_hub_download is None:
            error_msg = "huggingface-hub is required for HF_MODEL_ID but not installed."
            print(f"✗ {error_msg}", flush=True)
            raise ImportError(error_msg)
        
        # HF_FILENAME이 없으면 기본값 사용 (TinyLlama 작은 버전)
        if not hf_filename:
            if "tinyllama" in hf_model_id.lower():
                hf_filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
                print(f"  Using default filename: {hf_filename}", flush=True)
            else:
                error_msg = f"HF_FILENAME must be specified for model {hf_model_id}"
                print(f"✗ {error_msg}", flush=True)
                print(f"  Example: HF_FILENAME=model.q4_k_m.gguf", flush=True)
                raise ValueError(error_msg)
        
        print(f"Downloading model from Hugging Face Hub: {hf_model_id}/{hf_filename}", flush=True)
        try:
            cache_dir = os.getenv('HF_CACHE_DIR', '/tmp/models')
            model_path = hf_hub_download(
                repo_id=hf_model_id,
                filename=hf_filename,
                cache_dir=cache_dir
            )
            print(f"✓ Model downloaded to: {model_path}", flush=True)
        except Exception as e:
            error_msg = f"Failed to download model from Hugging Face Hub: {str(e)}"
            print(f"✗ {error_msg}", flush=True)
            raise RuntimeError(error_msg) from e
    
    # 로컬 파일 경로 확인
    if model_path and not os.path.exists(model_path):
        error_msg = f"Model file not found at path: {model_path}"
        print(f"✗ {error_msg}", flush=True)
        raise FileNotFoundError(error_msg)
    
    # llama-cpp-python 설치 확인
    if Llama is None:
        error_msg = "llama-cpp-python is not installed. Install it with: pip install llama-cpp-python"
        print(f"✗ {error_msg}", flush=True)
        raise ImportError(error_msg)
    
    # 모델 로딩 (필수, 실패 시 서버 시작 중단)
    print(f"Loading LLaMA model from {model_path}...", flush=True)
    try:
        llama_model = Llama(
            model_path=model_path,
            n_ctx=512,  # 컨텍스트 크기
            n_threads=4,  # 스레드 수
            verbose=False
        )
        print(f"✓ LLaMA model loaded successfully from {model_path}", flush=True)
        print(f"✓ Server is ready", flush=True)
        print(f"Health Check: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/health", flush=True)
        print(f"{'='*60}\n", flush=True)
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        print(f"✗ {error_msg}", flush=True)
        raise RuntimeError(error_msg) from e
    
    yield
    
    # Shutdown
    llama_model = None
    print("Shutting down server...", flush=True)


app = FastAPI(lifespan=lifespan)

@app.get("/")
def greet_json():
    """루트 엔드포인트"""
    return {
        "service": "LLaMA.cpp Server",
        "version": "2.1.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    """헬스체크 엔드포인트"""
    model_status = {
        "loaded": llama_model is not None,
        "type": "llama-cpp-python" if llama_model is not None else None
    }
    
    # 모델이 로드되어 있으면 간단한 질문에 대한 응답 생성
    sample_question = "Hello"
    sample_response = None
    
    if llama_model is not None:
        try:
            # 간단한 텍스트 생성
            output = llama_model(
                sample_question,
                max_tokens=20,
                temperature=0.7,
                top_p=0.9,
                echo=False,
                stop=["\n"]
            )
            # llama-cpp-python은 dict-like 객체를 반환
            if hasattr(output, 'choices'):
                sample_response = output.choices[0].text.strip()
            else:
                sample_response = output['choices'][0]['text'].strip()
        except Exception as e:
            sample_response = f"Error: {str(e)}"
    
    return {
        "status": "healthy",
        "service": "LLaMA.cpp Server",
        "version": "2.1.0",
        "model": model_status,
        "sample": {
            "question": sample_question if llama_model is not None else None,
            "response": sample_response
        }
    }

@app.post("/generate")
def generate_text(request: GenerateRequest):
    """텍스트 생성 엔드포인트"""
    if llama_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set MODEL_PATH environment variable.")
    
    try:
        # LLaMA 모델로 텍스트 생성
        output = llama_model(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            echo=False
        )
        
        # llama-cpp-python은 dict-like 객체를 반환
        if hasattr(output, 'choices'):
            generated_text = output.choices[0].text.strip()
        else:
            generated_text = output['choices'][0]['text'].strip()
        
        return {
            "prompt": request.prompt,
            "generated_text": generated_text,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.post("/completion")
def completion(request: GenerateRequest):
    """OpenAI 호환 completion 엔드포인트"""
    if llama_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set MODEL_PATH environment variable.")
    
    try:
        output = llama_model(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            echo=False
        )
        
        # llama-cpp-python은 dict-like 객체를 반환
        if hasattr(output, 'choices'):
            generated_text = output.choices[0].text.strip()
        else:
            generated_text = output['choices'][0]['text'].strip()
        
        # usage 정보는 llama-cpp-python에서 제공하지 않을 수 있음
        usage_info = {}
        if hasattr(output, 'usage'):
            usage_info = {
                "prompt_tokens": getattr(output.usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(output.usage, 'completion_tokens', 0),
                "total_tokens": getattr(output.usage, 'total_tokens', 0)
            }
        elif isinstance(output, dict) and 'usage' in output:
            usage_info = output['usage']
        else:
            usage_info = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        
        return {
            "id": "llama-cpp-completion",
            "object": "text_completion",
            "created": 0,
            "model": "llama-cpp",
            "choices": [
                {
                    "text": generated_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": usage_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Completion error: {str(e)}")


#푸쉬용 임시 주석
# 

