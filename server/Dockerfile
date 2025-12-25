# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile
# GGUF usage: https://huggingface.co/docs/hub/en/gguf-llamacpp

FROM python:3.11-slim

# llama-cpp-python 빌드에 필요한 의존성 설치
# 빌드 도구를 root에서 설치 (USER 전에)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    make \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# llama-cpp-python 빌드를 위한 환경변수 설정
# 허깅페이스 Spaces에서 안정적인 빌드를 위한 최적화
ENV FORCE_CMAKE=1
ENV CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_AVX=ON -DGGML_AVX2=ON"
# 빌드 병렬화 (빌드 시간 단축)
ENV CMAKE_BUILD_PARALLEL_LEVEL=2

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt

# pip 및 빌드 도구 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 먼저 다른 의존성 설치 (빌드 시간 단축)
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# llama-cpp-python을 마지막에 설치 (빌드 최적화)
# 허깅페이스 Spaces에서 빌드 타임아웃을 피하기 위해 최적화된 빌드
# 빌드 실패 시 재시도 로직 포함
RUN pip install --no-cache-dir llama-cpp-python[server]>=0.2.0 || \
    (echo "First attempt failed, retrying with verbose output..." && \
     pip install --no-cache-dir --verbose llama-cpp-python[server]>=0.2.0)

COPY --chown=user . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
