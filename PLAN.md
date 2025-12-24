---
name: 허깅페이스 Docker 배포 계획
overview: 허깅페이스 Spaces에 Docker 이미지를 빌드하고 푸시하는 프로젝트. 먼저 간단한 헬스체크 서버를 프레임워크 없이 구현하고, 이후 gpt-visualizer의 전체 구성을 배포합니다.
todos:
  - id: create-health-server
    content: 간단한 헬스체크 서버 구현 (server.py) - Python http.server 사용, /health 엔드포인트, JSON 응답
    status: completed
  - id: create-dockerfile
    content: Dockerfile 생성 - Python 3.11-slim, 포트 7860, server.py 실행
    status: completed
    dependencies:
      - create-health-server
  - id: create-github-actions
    content: GitHub Actions 워크플로우 생성 - Docker 빌드 및 허깅페이스 Spaces에 푸시
    status: completed
    dependencies:
      - create-dockerfile
  - id: create-readme
    content: README.md 작성 - 프로젝트 설명, 실행 방법, 배포 가이드
    status: completed
  - id: create-dockerignore
    content: .dockerignore 파일 생성 - 불필요한 파일 제외
    status: completed
---

# 허깅페이스 Docker Spaces 배포 계획

## 프로젝트 개요

이 프로젝트는 허깅페이스 Spaces에 Docker 기반 이미지를 빌드하고 푸시하는 것을 테스트하기 위한 프로젝트입니다. 단계적으로 진행하여 먼저 간단한 헬스체크 서버를 배포하고, 이후 상위 폴더의 `gpt-visualizer` 전체 구성을 배포합니다.

**허깅페이스 Space**: [JaceDashS/test](https://huggingface.co/spaces/JaceDashS/test)

**참고**: 허깅페이스 Spaces는 Git 저장소로 동작하며, Dockerfile을 포함한 파일들을 푸시하면 자동으로 Docker 이미지를 빌드하고 배포합니다. 포트는 반드시 7860을 사용해야 합니다.

## 단계 1: 간단한 헬스체크 서버 ✅

### 1.1 Python 헬스체크 서버 구현

- **파일**: `server.py`
- Python 표준 라이브러리의 `http.server` 모듈만 사용 (프레임워크 없이)
- `/health` 엔드포인트 구현
- JSON 응답 반환: `{"status": "healthy", "service": "Health Check Server", "version": "1.0.0"}`
- 포트는 환경변수 `PORT`로 설정 (기본값: 7860, 허깅페이스 Spaces 기본 포트)

### 1.2 Dockerfile 생성

- **파일**: `Dockerfile`
- Python 3.11-slim 베이스 이미지 사용
- `server.py` 복사 및 실행
- 포트 7860 노출
- CMD로 `python -u server.py` 실행 (unbuffered 모드)

### 1.3 GitHub Actions 워크플로우

- **파일**: `.github/workflows/docker-push.yml`
- Docker 이미지 빌드
- 허깅페이스 Spaces에 Git 푸시
- `HF_TOKEN` 시크릿 사용 (이미 설정됨)
- Space 이름: `JaceDashS/test` (https://huggingface.co/spaces/JaceDashS/test)
- 참고: 허깅페이스 Spaces는 Git 저장소로도 동작하므로, Dockerfile을 포함한 파일들을 Git으로 푸시하면 자동으로 빌드됨

### 1.4 README 작성

- **파일**: `README.md`
- 프로젝트 설명
- 로컬 실행 방법
- Docker 빌드 및 실행 방법
- 허깅페이스 Spaces 배포 방법

## 단계 2: server/ 폴더 구조로 전환

### 2.1 파일 이동

- 단계 1에서 생성한 파일들을 `server/` 폴더로 이동
- `server.py` → `server/main.py` (또는 `server/server.py`)
- `Dockerfile` → `server/Dockerfile`
- `README.md` → `server/README.md`
- `.dockerignore` → `server/.dockerignore`

### 2.2 Dockerfile 수정

- `server/Dockerfile`에서 파일 경로 수정
- `COPY server.py` → `COPY main.py` (또는 파일명에 맞게 수정)
- `CMD ["python", "-u", "server.py"]` → `CMD ["python", "-u", "main.py"]` (또는 파일명에 맞게 수정)

### 2.3 GitHub Actions 워크플로우 업데이트

- 허깅페이스 Spaces에 푸시할 때 `server/` 폴더를 루트로 인식하도록 수정
- 또는 루트의 Dockerfile이 `server/` 폴더를 빌드 컨텍스트로 사용하도록 설정

## 단계 3: llama-cpp-python 서버 프레임워크 적용

### 3.1 llama-cpp-python 서버 프레임워크 도입

- `llama-cpp-python`의 내장 서버 기능 활용
- `llama_cpp.server` 모듈 사용
- 기존 `http.server` 기반 코드를 `llama_cpp.server`로 마이그레이션
- 더 효율적인 모델 서빙 및 API 엔드포인트 제공

### 3.2 서버 구조 변경

- `server/main.py`: `llama_cpp.server` 기반으로 재작성
- `llama_cpp.server`의 내장 엔드포인트 활용 또는 커스텀 엔드포인트 추가
- 기존 라우트 로직을 `llama_cpp.server` 프레임워크에 맞게 조정

### 3.3 Dockerfile 및 의존성 업데이트

- `llama-cpp-python` 최신 버전 사용
- `llama_cpp.server` 관련 의존성 확인 및 추가
- 서버 시작 명령어 변경 (필요시)

## 단계 4: 런타임 모델 로딩 (gpt-visualizer 스타일)

### 4.1 모델 다운로드 및 로딩 구현

- 상위 폴더의 `../gpt-visualizer/server/` 구조 참조
- `server/model.py`: 런타임에 Hugging Face Hub에서 모델 다운로드
- 모델은 빌드 타임이 아닌 **런타임에 다운로드** (서버 시작 시)
- `llama-cpp-python`을 사용한 모델 로딩
- `server/requirements.txt`에 의존성 추가: `llama-cpp-python`, `huggingface-hub`

### 4.2 서버 파일 확장

- `server/main.py`: 메인 서버 파일 (기존 유지)
- `server/routes.py`: 라우트 핸들러 추가
- `server/model.py`: 런타임 모델 다운로드 및 로딩 로직
- `server/utils.py`: 유틸리티 함수 (PCA, 임베딩 처리 등)
- `server/schemas.py`: 데이터 스키마
- `server/config.py`: 설정 파일

### 4.3 Dockerfile 수정

- 빌드 타임 모델 다운로드 제거 (런타임 다운로드로 변경)
- `llama-cpp-python` 빌드 의존성 추가 (gcc, g++, cmake 등)
- `server/requirements.txt` 설치
- `llama_cpp.server` 기반 구조 유지

### 4.4 복잡한 의존성 처리

- `llama-cpp-python` 빌드 의존성
- Hugging Face Hub 모델 다운로드 (런타임)
- PCA 및 임베딩 처리
- `server/requirements.txt`에 모든 의존성 정의

## 파일 구조

### 단계 1 구조 (간단한 헬스체크)

```
hf-docker-space-cicd-test/
├── .github/
│   └── workflows/
│       └── docker-push.yml      # GitHub Actions 워크플로우
├── server.py                     # 간단한 헬스체크 서버
├── Dockerfile                   # Docker 이미지 빌드 파일
├── README.md                    # 프로젝트 문서
└── .dockerignore                # Docker 빌드 컨텍스트 제외 파일
```

### 단계 2 구조 (server/ 폴더로 이동)

```
hf-docker-space-cicd-test/
├── .github/
│   └── workflows/
│       └── docker-push.yml      # GitHub Actions 워크플로우
└── server/                      # 서버 폴더
    ├── Dockerfile               # Docker 이미지 빌드 파일
    ├── README.md                # 서버 문서
    ├── main.py                  # 서버 메인 파일 (server.py에서 이동)
    └── .dockerignore            # Docker 빌드 컨텍스트 제외 파일
```

### 단계 3 구조 (llama-cpp-python 서버 프레임워크)

```
hf-docker-space-cicd-test/
├── .github/
│   └── workflows/
│       └── docker-push.yml      # GitHub Actions 워크플로우
└── server/                      # 서버 폴더
    ├── Dockerfile               # Docker 이미지 빌드 파일
    ├── README.md                # 서버 문서
    ├── main.py                  # llama_cpp.server 기반 서버
    ├── routes.py                # 라우트 핸들러 (llama_cpp.server용)
    ├── requirements.txt         # Python 의존성 (llama-cpp-python 포함)
    └── .dockerignore            # Docker 빌드 컨텍스트 제외 파일
```

### 단계 4 구조 (런타임 모델 로딩)

```
hf-docker-space-cicd-test/
├── .github/
│   └── workflows/
│       └── docker-push.yml      # GitHub Actions 워크플로우
└── server/                      # 서버 폴더
    ├── Dockerfile               # Docker 이미지 빌드 파일
    ├── README.md                # 서버 문서
    ├── main.py                  # llama_cpp.server 기반 서버
    ├── routes.py                # 라우트 핸들러
    ├── model.py                 # 런타임 모델 로딩
    ├── utils.py                 # 유틸리티 함수
    ├── schemas.py               # 데이터 스키마
    ├── config.py                # 설정 파일
    ├── requirements.txt         # Python 의존성
    └── .dockerignore            # Docker 빌드 컨텍스트 제외 파일
```

## 기술 스택

### 단계 1-2
- **언어**: Python 3.11
- **서버**: Python 표준 라이브러리 `http.server` (프레임워크 없이)

### 단계 3
- **언어**: Python 3.11
- **서버**: `llama_cpp.server` 프레임워크
- **의존성**: `llama-cpp-python` 등

### 단계 4
- **언어**: Python 3.11
- **서버**: `llama_cpp.server` 프레임워크
- **모델**: `llama-cpp-python` (런타임 다운로드)
- **의존성**: `huggingface-hub`, `numpy`, `scikit-learn` 등

### 공통
- **컨테이너**: Docker
- **배포**: Hugging Face Spaces (Docker 타입)
- **CI/CD**: GitHub Actions

## 주요 구현 사항

### server.py 구조

```python
- HTTPRequestHandler 클래스 구현
- /health GET 엔드포인트
- CORS 헤더 지원
- JSON 응답 포맷
- 환경변수 PORT 읽기
```

### Dockerfile 구조

**단계 1 (루트 Dockerfile)**:

```dockerfile
- FROM python:3.11-slim
- WORKDIR /app
- COPY server.py
- EXPOSE 7860
- ENV PORT=7860
- CMD ["python", "-u", "server.py"]
```

**단계 2 (server/Dockerfile - 파일 이동 후)**:

```dockerfile
- FROM python:3.11-slim
- WORKDIR /app
- COPY main.py
- EXPOSE 7860
- ENV PORT=7860
- CMD ["python", "-u", "main.py"]
```

**단계 3 (server/Dockerfile - llama-cpp-python 서버 프레임워크)**:

```dockerfile
- FROM python:3.11-slim
- WORKDIR /app
- # 빌드 의존성 (llama-cpp-python 빌드용)
- RUN apt-get update && apt-get install -y gcc g++ cmake
- COPY requirements.txt
- RUN pip install --no-cache-dir -r requirements.txt
- COPY . /app
- EXPOSE 7860
- ENV PORT=7860
- # llama_cpp.server 사용
- CMD ["python", "-u", "main.py"]
```

**단계 4 (server/Dockerfile - 런타임 모델 로딩)**:

```dockerfile
- FROM python:3.11-slim
- WORKDIR /app
- # 빌드 의존성 (llama-cpp-python 빌드용)
- RUN apt-get update && apt-get install -y gcc g++ cmake
- COPY requirements.txt
- RUN pip install --no-cache-dir -r requirements.txt
- COPY . /app
- EXPOSE 7860
- ENV PORT=7860
- # llama_cpp.server + 런타임 모델 로딩
- CMD ["python", "-u", "main.py"]
```

### GitHub Actions 워크플로우

- 트리거: push to main/master
- 단계:
  1. 허깅페이스 CLI 설치 및 로그인
  2. Space 저장소 클론 또는 파일 푸시
  3. Dockerfile 및 관련 파일들을 허깅페이스 Spaces 저장소에 푸시
  4. 허깅페이스 Spaces가 자동으로 Docker 이미지를 빌드하고 배포

**참고**: 허깅페이스 Spaces는 Git 저장소로 동작하므로, Dockerfile을 포함한 파일들을 직접 푸시하면 자동으로 빌드됩니다. GitHub Actions를 통해 자동화하거나, 수동으로 Git 푸시도 가능합니다.

## 다음 단계

1. **단계 1**: ✅ 간단한 헬스체크 서버 구현 및 허깅페이스 Spaces 배포 테스트 (완료)
2. **단계 2**: 파일들을 `server/` 폴더로 이동하고 Dockerfile 수정
3. **단계 3**: `llama_cpp.server` 프레임워크 적용
4. **단계 4**: 런타임 모델 로딩 구현 (gpt-visualizer 스타일)

