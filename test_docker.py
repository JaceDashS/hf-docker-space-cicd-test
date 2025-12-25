#!/usr/bin/env python3
"""
Docker 컨테이너 테스트 스크립트
로컬 Docker 컨테이너의 embedding 엔드포인트를 테스트합니다.
"""
import json
import sys
import requests
import time
from typing import Optional

# UnicodeEncodeError 방지를 위해 stdout 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

DOCKER_URL = "http://localhost:7860"

# 고정 답변을 얻을 수 있는 명확한 프롬프트
FIXED_PROMPT = "What is 2+2? Answer with only the number."

def test_embedding_endpoint(url: str = DOCKER_URL, input_text: str = FIXED_PROMPT):
    """Embedding 엔드포인트 테스트"""
    print(f"Input text: {input_text}")
    print("-" * 60)
    
    try:
        print("요청 전송 중...")
        start_time = time.time()
        response = requests.post(
            f"{url}/embedding",
            headers={"Content-Type": "application/json"},
            json={"input_text": input_text},
            timeout=120  # 임베딩 추출은 시간이 걸릴 수 있음
        )
        elapsed_time = time.time() - start_time
        
        response.raise_for_status()
        
        data = response.json()
        
        # 응답 출력
        print(f"\n✓ 응답: {data.get('response', 'N/A')}")
        print(f"✓ 응답 시간: {elapsed_time:.3f}초")
        print()
        
        # 토큰별 임베딩 출력
        tokens = data.get("tokens", [])
        if tokens:
            print(f"✓ 토큰 임베딩 (총 {len(tokens)}개):")
            for token_data in tokens:
                token = token_data.get("token", "")
                embedding = token_data.get("embedding", [])
                dim = token_data.get("dim", 0)
                
                # 임베딩을 문자열로 변환 (앞 3개 + ... 형식)
                if isinstance(embedding, list):
                    # "..." 문자열이 마지막에 있으면 그대로 사용
                    if len(embedding) > 0 and embedding[-1] == "...":
                        # 앞 3개 숫자만 추출하여 포맷팅
                        numeric_values = [e for e in embedding if isinstance(e, (int, float))]
                        if len(numeric_values) >= 3:
                            embedding_str = f"[{numeric_values[0]}, {numeric_values[1]}, {numeric_values[2]}, ...]"
                        else:
                            embedding_str = str(embedding)
                    else:
                        # 숫자만 있는 경우 앞 3개만 표시
                        if len(embedding) > 3:
                            embedding_str = f"[{embedding[0]}, {embedding[1]}, {embedding[2]}, ...]"
                        else:
                            embedding_str = str(embedding)
                else:
                    embedding_str = str(embedding)
                
                # 실제 토큰 문자열 표시 (토큰이 비어있으면 "N/A" 표시)
                token_display = token if token else "N/A"
                print(f"  {token_display}: {embedding_str} (dim={dim})")
        else:
            print("  (토큰 임베딩 없음)")
        
        return True
        
    except requests.exceptions.Timeout:
        print("✗ 타임아웃: 서버가 응답하는 데 너무 오래 걸렸습니다.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Embedding 엔드포인트 테스트 실패: {e}")
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            print(f"  Response status: {status_code}")
            try:
                print(f"  Response body: {e.response.text[:200]}")
            except:
                pass
            
            # 404 에러인 경우 특별한 안내
            if status_code == 404:
                print("\n  ⚠ 404 에러: /embedding 엔드포인트를 찾을 수 없습니다.")
                print("  가능한 원인:")
                print("    1. 컨테이너가 최신 코드로 빌드되지 않았습니다.")
                print("    2. 서버 코드에 /embedding 엔드포인트가 없습니다.")
                print("\n  해결 방법:")
                print("    npm run docker:stop")
                print("    npm run docker:build:run")
                print("    python test_docker.py")
        return False
    except Exception as e:
        print(f"✗ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("Docker 컨테이너 Embedding 엔드포인트 테스트")
    print("=" * 60)
    print(f"테스트 URL: {DOCKER_URL}")
    print("\n참고: 컨테이너가 실행 중이어야 합니다.")
    print("      실행: npm run docker:build:run")
    print()
    
    # 명령줄 인자로 input_text를 받을 수 있음 (기본값은 FIXED_PROMPT)
    input_text = sys.argv[1] if len(sys.argv) > 1 else FIXED_PROMPT
    
    # Embedding 엔드포인트 테스트
    embedding_ok = test_embedding_endpoint(input_text=input_text)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    print(f"Embedding 엔드포인트: {'✓ 통과' if embedding_ok else '✗ 실패'}")
    
    if embedding_ok:
        print("\n✓ 테스트 통과!")
        sys.exit(0)
    else:
        print("\n✗ 테스트 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()
