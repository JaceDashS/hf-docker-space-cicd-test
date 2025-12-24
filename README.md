---
title: Health Check Server
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "latest"
pinned: false
---

# 허깅페이스 Docker Spaces 배포 테스트

허깅페이스 Spaces에 Docker 기반 이미지를 빌드하고 푸시하는 것을 테스트하기 위한 프로젝트입니다.

**허깅페이스 Space**: [JaceDashS/test](https://huggingface.co/spaces/JaceDashS/test)

## 프로젝트 개요

이 프로젝트는 단계적으로 진행됩니다:

1. **단계 1**: 간단한 헬스체크 서버 (현재 단계)
2. **단계 2**: `server/` 폴더 구조로 전환
3. **단계 3**: 런타임 모델 로딩 (gpt-visualizer 스타일)
4. **단계 4**: `llama_cpp.server` 프레임워크 적용

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
  "version": "1.0.0"
}
```

## 로컬 실행 방법

### npm 스크립트 사용 (권장)

```bash
# 서버 실행
npm start
# 또는
npm run dev

# Docker 빌드
npm run docker:build

# Docker 빌드 (캐시 없이)
npm run docker:build:no-cache

# Docker 실행
npm run docker:run

# Docker 빌드 + 실행 (한 번에)
npm run docker:build:run

# Docker 로그 확인
npm run docker:logs

# Docker 중지
npm run docker:stop

# 헬스체크 테스트
npm test
# 또는 JSON 포맷으로 확인
npm run health
```

### Python으로 직접 실행

```bash
# 기본 포트(7860)로 실행
python server.py

# 또는 포트 지정
PORT=8000 python server.py
```

### Docker로 직접 실행

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

### Space 생성

1. [허깅페이스 Spaces](https://huggingface.co/spaces)에 접속
2. "Create new Space" 클릭
3. Space 설정:
   - **Space name**: 원하는 이름 입력 (예: `test`)
   - **SDK**: **Docker** 선택
   - **Visibility**: Public 또는 Private 선택
4. Space 생성 완료

### GitHub Secrets 설정

GitHub Actions를 사용하려면 허깅페이스 토큰을 설정해야 합니다:

1. [허깅페이스 설정 페이지](https://huggingface.co/settings/tokens)에서 토큰 생성
   - **Write** 권한이 있는 토큰 생성
2. GitHub 저장소 설정:
   - 저장소 → Settings → Secrets and variables → Actions
   - "New repository secret" 클릭
   - **Name**: `HF_TOKEN`
   - **Value**: 생성한 허깅페이스 토큰 입력
   - "Add secret" 클릭

### Space 저장소 정보

- **Space URL**: https://huggingface.co/spaces/JaceDashS/test
- **Git 저장소**: https://huggingface.co/spaces/JaceDashS/test
- **Space 이름**: `JaceDashS/test`

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
cp ../server.py .
cp ../Dockerfile .

# 커밋 및 푸시
git add server.py Dockerfile
git commit -m "Add health check server"
git push
```

## 기술 스택

- **언어**: Python 3.11
- **서버**: Python 표준 라이브러리 `http.server` (프레임워크 없이)
- **컨테이너**: Docker
- **배포**: Hugging Face Spaces (Docker 타입)
- **CI/CD**: GitHub Actions
- **스크립트 관리**: npm (편의를 위한 스크립트)

## 참고사항

- 허깅페이스 Spaces는 Git 저장소로 동작합니다
- Dockerfile을 포함한 파일들을 푸시하면 자동으로 Docker 이미지를 빌드하고 배포합니다
- 포트는 반드시 7860을 사용해야 합니다
- 환경변수 `PORT`와 `HOST`를 통해 설정 가능합니다

## 다음 단계

- [ ] 단계 2: 파일들을 `server/` 폴더로 이동
- [ ] 단계 3: 런타임 모델 로딩 구현
- [ ] 단계 4: `llama_cpp.server` 프레임워크 적용

