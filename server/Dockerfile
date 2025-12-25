# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.11-slim

# llama-cpp-python 빌드에 필요한 의존성 설치
# 빌드 도구를 root에서 설치 (USER 전에)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    make \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# llama-cpp-python 빌드를 위한 환경변수 설정
# FORCE_CMAKE=1: CMake 빌드 강제 사용 (허깅페이스 Spaces에서 안정적)
# CMAKE_ARGS: 빌드 최적화 옵션
# CPU 전용 빌드 (GPU 없이)
ENV FORCE_CMAKE=1
ENV CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt

# llama-cpp-python 빌드 최적화를 위해 pip 업그레이드 및 빌드 설정
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
