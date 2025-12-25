#!/usr/bin/env python3
"""
임베딩 엔드포인트 테스트 스크립트
포맷팅된 JSON 응답을 출력합니다.
"""
import json
import sys
import requests
from typing import Optional

def test_embedding(
    url: str = "https://jacedashs-test.hf.space/embedding",
    input_text: str = "Hello, who are you?"
):
    """임베딩 엔드포인트를 테스트하고 포맷팅된 응답을 출력"""
    try:
        print(f"Testing embedding endpoint: {url}")
        print(f"Input text: {input_text}")
        print("-" * 60)
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"input_text": input_text},
            timeout=120  # 임베딩 추출은 시간이 걸릴 수 있으므로 타임아웃 증가
        )
        
        response.raise_for_status()
        
        # JSON 응답 파싱
        data = response.json()
        
        # 사용자 요청 형식으로 출력
        print("\n응답:", data.get("response", ""))
        print()
        
        # 토큰별 임베딩 출력
        tokens = data.get("tokens", [])
        for i, token_data in enumerate(tokens, 1):
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
            
            print(f"토큰{i}: {embedding_str} (dim={dim})")
        
        print()
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}", file=sys.stderr)
            try:
                print(f"Response body: {e.response.text}", file=sys.stderr)
            except:
                pass
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import os
    
    # 명령줄 인자로 input_text를 받을 수 있음
    input_text = sys.argv[1] if len(sys.argv) > 1 else "Hello, who are you?"
    
    # URL도 환경변수나 인자로 받을 수 있음
    # 로컬 서버 테스트: EMBEDDING_URL=http://localhost:7860/embedding python test_embedding.py
    url = os.getenv("EMBEDDING_URL", "https://jacedashs-test.hf.space/embedding")
    
    # 두 번째 인자로 URL을 지정할 수도 있음
    if len(sys.argv) > 2:
        url = sys.argv[2]
    
    test_embedding(url=url, input_text=input_text)

