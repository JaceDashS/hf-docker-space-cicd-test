#!/usr/bin/env python3
"""
클라우드 스레드별 성능 벤치마크 스크립트
각 스레드(1~8)에 대해 클라우드의 test_embedding.py를 5번 실행하고 결과를 수집합니다.
사용자가 클라우드를 재배포하면서 스레드를 바꿀 때마다 실행합니다.
"""
import subprocess
import sys
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Optional

# UnicodeEncodeError 방지를 위해 stdout 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# 클라우드 URL (허깅페이스 Spaces)
CLOUD_URL = "https://jacedashs-test.hf.space"

def run_test_embedding(run_number: int) -> Optional[float]:
    """test_embedding.py를 실행하고 응답 시간을 반환 (클라우드)"""
    try:
        print(f"  [실행 {run_number}/5] test_embedding.py 실행 중...")
        
        # test_embedding.py 실행 (클라우드 URL 사용)
        start_time = time.time()
        env = os.environ.copy()
        # 클라우드 URL 사용 (기본값)
        if "EMBEDDING_URL" not in env:
            env["EMBEDDING_URL"] = CLOUD_URL
        
        result = subprocess.run(
            [sys.executable, "test_embedding.py"],
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=600
        )
        elapsed_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"    ✗ 테스트 실패: {result.stderr[:200] if result.stderr else 'Unknown error'}")
            return None
        
        # 출력에서 응답 시간 추출
        output_lines = result.stdout.split('\n')
        response_time = None
        import re
        for line in output_lines:
            # "✓ 응답 시간: 68.852초" 형식에서 숫자 추출
            if '응답 시간:' in line:
                match = re.search(r'응답 시간:\s*(\d+\.?\d*)', line)
                if match:
                    response_time = float(match.group(1))
                    break
        
        if response_time is None:
            # 응답 시간을 찾을 수 없으면 전체 실행 시간 사용
            print(f"    ⚠ 출력에서 응답 시간을 찾을 수 없어 전체 실행 시간 사용: {elapsed_time:.3f}초")
            response_time = elapsed_time
        
        print(f"    ✓ 응답 시간: {response_time:.3f}초")
        return response_time
        
    except subprocess.TimeoutExpired:
        print(f"    ✗ 타임아웃")
        return None
    except Exception as e:
        print(f"    ✗ 오류: {e}")
        return None

def append_to_markdown(n_threads: int, results: List[float]):
    """결과를 마크다운 파일에 append"""
    md_file = "benchmark_results.md"
    
    # 파일이 없으면 헤더 작성
    if not os.path.exists(md_file):
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"""# 스레드별 성능 벤치마크 결과

생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 테스트 설정
- 테스트 스크립트: `test_embedding.py`
- 실행 횟수: 각 스레드당 5회
- 테스트 URL: {CLOUD_URL}
- 프롬프트: "What is 2+2? Answer with only the number."

## 결과 표

| 스레드 | 실행 1 (초) | 실행 2 (초) | 실행 3 (초) | 실행 4 (초) | 실행 5 (초) | 평균 시간 (초) |
|--------|-------------|-------------|-------------|-------------|-------------|----------------|
""")
    
    # 기존 파일 읽기
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 해당 스레드 행이 이미 있는지 확인
    lines = content.split('\n')
    header_end = -1
    for i, line in enumerate(lines):
        if '| 스레드 |' in line:
            header_end = i + 1
            break
    
    # 스레드 행 찾기 또는 추가할 위치 찾기
    thread_row_idx = -1
    for i in range(header_end, len(lines)):
        if lines[i].startswith(f"| {n_threads} |"):
            thread_row_idx = i
            break
    
    # 결과 행 생성
    avg_time = sum(results) / len(results) if results else 0
    row = [f"{n_threads}"]
    for i in range(5):
        if i < len(results):
            row.append(f"{results[i]:.3f}")
        else:
            row.append("-")
    row.append(f"{avg_time:.3f}" if results else "-")
    new_row = "| " + " | ".join(row) + " |\n"
    
    # 행 업데이트 또는 추가
    if thread_row_idx >= 0:
        # 기존 행 교체
        lines[thread_row_idx] = new_row.rstrip('\n')
    else:
        # 새 행 추가 (헤더 다음에)
        if header_end >= 0:
            lines.insert(header_end, new_row.rstrip('\n'))
        else:
            lines.append(new_row.rstrip('\n'))
    
    # 요약 섹션 업데이트
    summary_start = -1
    summary_end = -1
    for i, line in enumerate(lines):
        if line.strip() == "## 요약":
            summary_start = i
            break
    
    if summary_start >= 0:
        # 요약 섹션 찾기
        for i in range(summary_start + 1, len(lines)):
            if lines[i].strip().startswith("##"):
                summary_end = i
                break
        if summary_end == -1:
            summary_end = len(lines)
        
        # 모든 스레드 결과 수집
        all_results = {}
        for i in range(header_end, summary_start):
            if '|' in lines[i] and not lines[i].strip().startswith('|--'):
                parts = [p.strip() for p in lines[i].split('|')]
                if len(parts) >= 7 and parts[1].isdigit():
                    thread_num = int(parts[1])
                    times = []
                    for j in range(2, 7):
                        if parts[j] != '-' and parts[j]:
                            try:
                                times.append(float(parts[j]))
                            except:
                                pass
                    if times:
                        all_results[thread_num] = times
        
        # 요약 생성
        summary_lines = ["## 요약", ""]
        if all_results:
            avg_times = {n: sum(times) / len(times) for n, times in all_results.items()}
            fastest_thread = min(avg_times.items(), key=lambda x: x[1])
            slowest_thread = max(avg_times.items(), key=lambda x: x[1])
            
            summary_lines.append(f"- **가장 빠른 스레드**: {fastest_thread[0]} (평균 {fastest_thread[1]:.3f}초)")
            summary_lines.append(f"- **가장 느린 스레드**: {slowest_thread[0]} (평균 {slowest_thread[1]:.3f}초)")
            summary_lines.append(f"- **성능 향상**: {slowest_thread[1] / fastest_thread[1]:.2f}x")
        else:
            summary_lines.append("(결과 없음)")
        
        # 요약 섹션 교체
        lines[summary_start:summary_end] = summary_lines
    
    # 파일 저장
    with open(md_file, "w", encoding="utf-8") as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ 결과가 {md_file}에 추가되었습니다.")

def main():
    """메인 벤치마크 함수"""
    print("=" * 70)
    print("클라우드 스레드별 성능 벤치마크")
    print("=" * 70)
    print(f"테스트 URL: {CLOUD_URL}")
    print("\n사용 방법:")
    print("1. 클라우드를 원하는 스레드 수로 재배포하세요")
    print("2. 이 스크립트를 실행하면 해당 스레드에 대해 5번 테스트합니다")
    print("3. 결과는 benchmark_results.md 파일에 자동으로 추가됩니다")
    print("\n주의: 각 스레드마다 클라우드를 재배포한 후 이 스크립트를 실행하세요.")
    print()
    
    # 스레드 수 입력 받기
    try:
        n_threads = int(input("테스트할 스레드 수를 입력하세요 (1-8): "))
        if n_threads < 1 or n_threads > 8:
            print("✗ 스레드 수는 1-8 사이여야 합니다.")
            sys.exit(1)
    except ValueError:
        print("✗ 올바른 숫자를 입력하세요.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n취소되었습니다.")
        sys.exit(0)
    
    print(f"\n{'=' * 70}")
    print(f"스레드 {n_threads} 테스트 시작")
    print(f"{'=' * 70}")
    print(f"\n클라우드가 스레드 {n_threads}로 배포되었는지 확인하세요.")
    input("준비되면 Enter를 누르세요...")
    
    thread_results = []
    
    # 5번 실행
    for run_num in range(1, 6):
        response_time = run_test_embedding(run_num)
        if response_time is not None:
            thread_results.append(response_time)
        if run_num < 5:
            time.sleep(2)  # 각 실행 사이에 짧은 대기
    
    if thread_results:
        avg_time = sum(thread_results) / len(thread_results)
        print(f"\n스레드 {n_threads} 결과:")
        print(f"  실행 횟수: {len(thread_results)}/5")
        print(f"  개별 시간: {[f'{t:.3f}' for t in thread_results]}")
        print(f"  평균 시간: {avg_time:.3f}초")
        
        # 마크다운 파일에 append
        append_to_markdown(n_threads, thread_results)
    else:
        print(f"\n스레드 {n_threads}: 모든 실행 실패")
    
    print(f"\n{'=' * 70}")
    print("벤치마크 완료!")
    print("=" * 70)
    print(f"\n다음 스레드를 테스트하려면 이 스크립트를 다시 실행하세요.")

if __name__ == "__main__":
    main()
