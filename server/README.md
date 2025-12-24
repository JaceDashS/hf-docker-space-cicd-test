

# 허깅페이스 Docker Spaces 배포 테스트

허깅페이스 Spaces에 Docker 기반 이미지를 빌드하고 푸시하는 것을 테스트하기 위한 프로젝트입니다.

**허깅페이스 Space**: [JaceDashS/test](https://huggingface.co/spaces/JaceDashS/test)

## 프로젝트 개요

이 프로젝트는 단계적으로 진행됩니다:

1. **단계 1**: 간단한 헬스체크 서버 ✅
2. **단계 2**: `server/` 폴더 구조로 전환 ✅
3. **단계 3**: `llama_cpp.server` 프레임워크 적용
4. **단계 4**: 런타임 모델 로딩 (gpt-visualizer 스타일)

## 단계 1: 간단한 헬스체크 서버

Python 표준 라이브러리 `http.server`를 사용하여 프레임워크 없이 구현한 간단한 헬스체크 서버입니다.

### 기능

- `/health` 엔드포인트 제공
- JSON 응답 반환
- CORS 헤더 지원
- 포트 7860에서 실행 (허깅페이스 Spaces 기본 포트)

### API 엔드포인트

#### `GET /health`

헬스체크 엔드포인트입니다.

**응답 예시:**
```json
{
  "status": "healthy",
  "service": "Health Check Server",
  "version": "1.0.1"
}
```

## 로컬 실행 방법

### Python으로 직접 실행

```bash
# 기본 포트(7860)로 실행
python main.py

# 또는 포트 지정
PORT=8000 python main.py
```

### Docker로 실행

```bash
# 이미지 빌드
docker build -t health-check-server .

# 컨테이너 실행
docker run -p 7860:7860 health-check-server

# 또는 포트 변경
docker run -p 8000:7860 -e PORT=7860 health-check-server
```

### 테스트

```bash
# 헬스체크 확인
curl http://localhost:7860/health

# 또는 브라우저에서
# http://localhost:7860/health
```

## 허깅페이스 Spaces 구성

자세한 구성 방법은 [허깅페이스 구성 마크다운](HUGGINGFACE_SETUP.md)을 참조하세요.

## 허깅페이스 Spaces 배포

### 자동 배포 (GitHub Actions)

이 프로젝트는 GitHub Actions를 통해 자동으로 허깅페이스 Spaces에 배포됩니다.

**필수 설정:**
- GitHub Secrets에 `HF_TOKEN` 설정 (위의 "GitHub Secrets 설정" 참조)

**배포 프로세스:**
1. 코드를 GitHub에 푸시 (`main` 또는 `master` 브랜치)
2. GitHub Actions가 자동으로 실행됨
3. 허깅페이스 Spaces 저장소에 파일 푸시
4. 허깅페이스 Spaces가 자동으로 Docker 이미지를 빌드하고 배포
5. 배포 완료 후 Space 페이지에서 확인 가능

### 수동 배포

```bash
# 허깅페이스 CLI 설치 (필요시)
pip install huggingface_hub[cli]

# 허깅페이스 로그인
huggingface-cli login

# Space 저장소 클론
git clone https://huggingface.co/spaces/JaceDashS/test
cd test

# 파일 복사
cp ../server/main.py .
cp ../server/Dockerfile .

# 커밋 및 푸시
git add main.py Dockerfile
git commit -m "Add health check server"
git push
```

## 기술 스택

- **언어**: Python 3.11
- **서버**: Python 표준 라이브러리 `http.server` (프레임워크 없이)
- **컨테이너**: Docker
- **배포**: Hugging Face Spaces (Docker 타입)
- **CI/CD**: GitHub Actions

## 참고사항

- 허깅페이스 Spaces는 Git 저장소로 동작합니다
- Dockerfile을 포함한 파일들을 푸시하면 자동으로 Docker 이미지를 빌드하고 배포합니다
- 포트는 반드시 7860을 사용해야 합니다
- 환경변수 `PORT`와 `HOST`를 통해 설정 가능합니다

## 다음 단계

- [ ] 단계 3: `llama_cpp.server` 프레임워크 적용
- [ ] 단계 4: 런타임 모델 로딩 구현

