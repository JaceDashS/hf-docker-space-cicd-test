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
model_info = {
    "name": None,
    "path": None,
    "repo_id": None,
    "filename": None
}


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class EmbeddingRequest(BaseModel):
    input_text: str


class TokenEmbedding(BaseModel):
    token: str
    embedding: list  # 앞 3개만 표시, 나머지는 ...
    dim: int


class EmbeddingResponse(BaseModel):
    response: str  # 모델이 생성한 실제 응답
    tokens: list[TokenEmbedding]  # 토큰별 임베딩 리스트


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 및 종료 이벤트 처리"""
    global llama_model, model_info
    
    # Startup
    port = int(os.getenv('PORT', '7860'))
    host = os.getenv('HOST', '0.0.0.0')
    
    # unbuffered 출력을 위해 sys.stdout.flush() 사용
    print(f"\n{'='*60}", flush=True)
    print("LLaMA.cpp Server Starting...", flush=True)
    print(f"Version: 2.3.1", flush=True)
    print(f"Host: {host}", flush=True)
    print(f"Port: {port}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 모델 파일 경로 확인 및 다운로드
    model_path = os.getenv('MODEL_PATH', None)
    hf_model_id = os.getenv('HF_MODEL_ID', None)
    hf_filename = os.getenv('HF_FILENAME', None)  # 선택적, GGUF 파일명 지정
    
    # 환경변수가 없으면 기본 모델 사용 (Llama-3.2-1B-Instruct)
    if not model_path and not hf_model_id:
        print("ℹ No MODEL_PATH or HF_MODEL_ID specified, using default model", flush=True)
        hf_model_id = "bartowski/Llama-3.2-1B-Instruct-GGUF"
        hf_filename = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        print(f"  Default model: {hf_model_id}/{hf_filename}", flush=True)
    
    # 모델 정보 저장
    model_info["repo_id"] = hf_model_id
    model_info["filename"] = hf_filename
    
    # Hugging Face Hub에서 모델 다운로드
    if hf_model_id:
        if hf_hub_download is None:
            error_msg = "huggingface-hub is required for HF_MODEL_ID but not installed."
            print(f"✗ {error_msg}", flush=True)
            raise ImportError(error_msg)
        
        # HF_FILENAME이 없으면 기본값 사용
        if not hf_filename:
            if "llama-3.2-1b-instruct" in hf_model_id.lower():
                hf_filename = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
                print(f"  Using default filename: {hf_filename}", flush=True)
            elif "tinyllama" in hf_model_id.lower():
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
            model_info["path"] = model_path
            # 모델 이름 추출 (파일명에서 확장자 제거)
            if hf_filename:
                model_info["name"] = hf_filename.replace(".gguf", "")
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
        # Llama-3.2 모델 설정 (gpt-visualizer 스타일)
        n_threads = int(os.getenv('LLAMA_N_THREADS', '1'))
        llama_model = Llama(
            model_path=model_path,
            n_ctx=4096,  # 컨텍스트 크기 (Llama-3.2에 맞게 증가)
            n_threads=n_threads,  # 스레드 수 (환경변수로 설정 가능)
            n_gpu_layers=0,  # CPU 전용
            chat_format="llama-3",  # Llama-3 채팅 포맷
            embedding=True,  # 임베딩 추출 활성화
            verbose=False
        )
        print(f"✓ LLaMA model loaded successfully from {model_path}", flush=True)
        # 모델 정보 업데이트
        model_info["path"] = model_path
        if not model_info["name"]:
            # 경로에서 파일명 추출
            model_name = Path(model_path).stem
            model_info["name"] = model_name
        print(f"✓ Loaded model: {model_info['name']}", flush=True)
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
        "version": "2.3.1",
        "status": "running"
    }

@app.get("/health")
def health_check():
    """헬스체크 엔드포인트"""
    model_status = {
        "loaded": llama_model is not None,
        "type": "llama-cpp-python" if llama_model is not None else None,
        "name": model_info["name"] if llama_model is not None else None,
        "repo_id": model_info["repo_id"] if llama_model is not None else None,
        "filename": model_info["filename"] if llama_model is not None else None
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
        "version": "2.3.1",
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


@app.post("/embedding", response_model=EmbeddingResponse)
def get_embedding(request: EmbeddingRequest):
    """임베딩 벡터 추출 엔드포인트 - 토큰별 임베딩과 모델 응답 반환"""
    if llama_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait for model to load.")
    
    if not request.input_text or not request.input_text.strip():
        raise HTTPException(status_code=400, detail="input_text is required and cannot be empty")
    
    try:
        # 1. 모델로 텍스트 생성 (응답 생성)
        print(f"[EMBEDDING] Generating response for: {request.input_text[:50]}...", flush=True)
        output = llama_model(
            request.input_text,
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["\n"]
        )
        
        # 생성된 응답 텍스트 추출
        if hasattr(output, 'choices'):
            generated_text = output.choices[0].text.strip()
        else:
            generated_text = output['choices'][0]['text'].strip()
        
        print(f"[EMBEDDING] Generated response: {generated_text[:50]}...", flush=True)
        
        # 2. 입력 텍스트를 토큰화
        print(f"[EMBEDDING] Tokenizing input text...", flush=True)
        input_tokens = llama_model.tokenize(request.input_text.encode('utf-8'))
        input_token_strs = [llama_model.detokenize([t]).decode('utf-8', errors='replace') for t in input_tokens]
        
        # 빈 토큰 제거
        filtered_tokens = [(token_str, token_id) for token_str, token_id in zip(input_token_strs, input_tokens) if token_str.strip()]
        input_token_strs = [t for t, _ in filtered_tokens]
        input_token_ids = [tid for _, tid in filtered_tokens]
        
        print(f"[EMBEDDING] Filtered tokens: {len(input_token_strs)}", flush=True)
        
        # 3. 성능 개선: 전체 토큰 시퀀스를 한 번에 처리하여 내부 상태 구축
        # eval() 또는 _eval() 메서드를 사용하여 전체 토큰 시퀀스를 처리하고
        # 내부 hidden states에서 각 토큰의 임베딩을 추출
        print(f"[EMBEDDING] Processing full token sequence (optimized, no warnings)...", flush=True)
        token_embeddings = []
        
        try:
            # 방법 1: eval() 메서드를 사용하여 전체 토큰 시퀀스 처리
            # 이렇게 하면 내부 상태가 구축되고, 각 토큰의 hidden state에 접근 가능
            if hasattr(llama_model, 'eval'):
                # 전체 토큰 시퀀스를 eval()로 처리
                llama_model.eval(input_tokens)
                print(f"[EMBEDDING] Full sequence processed with eval()", flush=True)
                
                # 내부 상태에서 각 토큰의 임베딩 추출 시도
                # llama-cpp-python의 내부 API를 사용
                if hasattr(llama_model, 'get_embeddings'):
                    # get_embeddings()가 토큰별 임베딩 리스트를 반환하는 경우
                    all_embeddings = llama_model.get_embeddings()
                    if all_embeddings is not None and len(all_embeddings) > 0:
                        # 각 토큰에 대해 해당하는 임베딩 추출
                        for i, token_str in enumerate(input_token_strs):
                            if i < len(all_embeddings):
                                token_embedding = all_embeddings[i]
                            else:
                                # 인덱스가 범위를 벗어나면 마지막 임베딩 사용
                                token_embedding = all_embeddings[-1]
                            
                            # numpy array일 수 있으므로 리스트로 변환
                            if hasattr(token_embedding, 'tolist'):
                                embedding_list = token_embedding.tolist()
                            elif isinstance(token_embedding, list):
                                embedding_list = token_embedding
                            else:
                                embedding_list = list(token_embedding)
                            
                            dim = len(embedding_list)
                            if dim > 3:
                                embedding_display = embedding_list[:3] + ["..."]
                            else:
                                embedding_display = embedding_list
                            
                            token_embeddings.append(TokenEmbedding(
                                token=token_str,
                                embedding=embedding_display,
                                dim=dim
                            ))
                        print(f"[EMBEDDING] Extracted {len(token_embeddings)} token embeddings from internal state", flush=True)
                    else:
                        raise AttributeError("get_embeddings() returned empty or None")
                elif hasattr(llama_model, '_ctx'):
                    # 내부 컨텍스트에서 직접 추출 시도
                    # llama-cpp-python의 내부 구조에 따라 다를 수 있음
                    print(f"[EMBEDDING] Attempting to extract from internal context...", flush=True)
                    # 폴백으로 전체 텍스트 임베딩을 각 토큰에 할당
                    full_embedding = llama_model.embed(request.input_text)
                    if hasattr(full_embedding, 'tolist'):
                        full_embedding_list = full_embedding.tolist()
                    elif isinstance(full_embedding, list):
                        full_embedding_list = full_embedding
                    else:
                        full_embedding_list = list(full_embedding)
                    
                    dim = len(full_embedding_list)
                    if dim > 3:
                        embedding_display = full_embedding_list[:3] + ["..."]
                    else:
                        embedding_display = full_embedding_list
                    
                    # 각 토큰에 전체 임베딩 할당 (임시 방법)
                    for token_str in input_token_strs:
                        token_embeddings.append(TokenEmbedding(
                            token=token_str,
                            embedding=embedding_display,
                            dim=dim
                        ))
                    print(f"[EMBEDDING] Used full text embedding for all tokens (fallback)", flush=True)
                else:
                    raise AttributeError("No method to extract token-level embeddings found")
            else:
                # eval() 메서드가 없는 경우: 전체 텍스트를 한 번 처리하고
                # 각 토큰을 전체 컨텍스트에 포함시켜 처리
                print(f"[EMBEDDING] eval() not available, using alternative method...", flush=True)
                # 전체 텍스트 임베딩을 먼저 생성
                full_embedding = llama_model.embed(request.input_text)
                
                # 각 토큰을 전체 텍스트의 일부로 포함시켜 처리
                # 토큰의 위치를 고려하여 임베딩 추출
                for i, token_str in enumerate(input_token_strs):
                    # 토큰 앞부분의 컨텍스트를 포함한 부분 문자열 생성
                    # 이렇게 하면 전체 컨텍스트를 고려한 임베딩을 얻을 수 있음
                    token_start_idx = request.input_text.find(token_str)
                    if token_start_idx >= 0:
                        # 토큰이 포함된 부분 문자열 (토큰 앞의 컨텍스트 포함)
                        context_text = request.input_text[:token_start_idx + len(token_str)]
                        token_embedding = llama_model.embed(context_text)
                    else:
                        # 토큰을 찾을 수 없으면 전체 텍스트 임베딩 사용
                        token_embedding = full_embedding
                    
                    # numpy array일 수 있으므로 리스트로 변환
                    if hasattr(token_embedding, 'tolist'):
                        embedding_list = token_embedding.tolist()
                    elif isinstance(token_embedding, list):
                        embedding_list = token_embedding
                    else:
                        embedding_list = list(token_embedding)
                    
                    dim = len(embedding_list)
                    if dim > 3:
                        embedding_display = embedding_list[:3] + ["..."]
                    else:
                        embedding_display = embedding_list
                    
                    token_embeddings.append(TokenEmbedding(
                        token=token_str,
                        embedding=embedding_display,
                        dim=dim
                    ))
                print(f"[EMBEDDING] Extracted {len(token_embeddings)} token embeddings using context-aware method", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to extract token embeddings using optimized method: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # 최종 폴백: 각 토큰을 개별 처리 (경고 발생하지만 동작함)
            print(f"[EMBEDDING] Falling back to individual token processing...", flush=True)
            for token_str in input_token_strs:
                token_embedding = llama_model.embed(token_str)
                
                if hasattr(token_embedding, 'tolist'):
                    embedding_list = token_embedding.tolist()
                elif isinstance(token_embedding, list):
                    embedding_list = token_embedding
                else:
                    embedding_list = list(token_embedding)
                
                dim = len(embedding_list)
                if dim > 3:
                    embedding_display = embedding_list[:3] + ["..."]
                else:
                    embedding_display = embedding_list
                
                token_embeddings.append(TokenEmbedding(
                    token=token_str,
                    embedding=embedding_display,
                    dim=dim
                ))
        
        print(f"[EMBEDDING] Extracted {len(token_embeddings)} token embeddings", flush=True)
        
        # 첫 번째 토큰만 반환
        first_token_only = token_embeddings[:1] if token_embeddings else []
        print(f"[EMBEDDING] Returning response and first token embedding only", flush=True)
        
        return EmbeddingResponse(
            response=generated_text,
            tokens=first_token_only
        )
    except Exception as e:
        print(f"[ERROR] Embedding extraction failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding extraction error: {str(e)}")


#푸쉬용 임시 주석
#push test