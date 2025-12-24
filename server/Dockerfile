FROM python:3.11-slim

WORKDIR /app

# 서버 파일 복사
COPY main.py .

# 포트 노출
EXPOSE 7860

# 환경변수 설정
ENV PORT=7860
ENV HOST=0.0.0.0

# 서버 실행 (unbuffered 모드로 즉시 출력)
CMD ["python", "-u", "main.py"]

