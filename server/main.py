"""
헬스체크 서버
FastAPI를 사용하여 구현
"""
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 및 종료 이벤트 처리"""
    # Startup
    port = int(os.getenv('PORT', '7860'))
    host = os.getenv('HOST', '0.0.0.0')
    
    # unbuffered 출력을 위해 sys.stdout.flush() 사용
    sys.stdout.write(f"\n{'='*60}\n")
    sys.stdout.write("Health Check Server Started\n")
    sys.stdout.write(f"Version: 1.0.2\n")
    sys.stdout.write(f"Host: {host}\n")
    sys.stdout.write(f"Port: {port}\n")
    sys.stdout.write(f"Health Check: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/health\n")
    sys.stdout.write(f"{'='*60}\n\n")
    sys.stdout.flush()
    
    yield
    
    # Shutdown (필요시 추가)
    # sys.stdout.write("Shutting down server...\n")
    # sys.stdout.flush()


app = FastAPI(lifespan=lifespan)

@app.get("/")
def greet_json():
    """루트 엔드포인트"""
    return {"Hello": "World!"}

@app.get("/health")
def health_check():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy",
        "service": "Health Check Server",
        "version": "1.0.2"
    }
