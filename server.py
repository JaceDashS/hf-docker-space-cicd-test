"""
간단한 헬스체크 서버
Python 표준 라이브러리 http.server를 사용하여 프레임워크 없이 구현
"""
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP 핸들러 - 헬스체크 엔드포인트 제공"""
    
    def _set_cors_headers(self):
        """CORS 헤더 설정"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def do_OPTIONS(self):
        """OPTIONS 요청 처리 (CORS preflight)"""
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        """GET 요청 처리"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            self._handle_health()
        else:
            self._send_error(404, "Not Found")
    
    def _handle_health(self):
        """헬스체크 엔드포인트 처리"""
        response = {
            "status": "healthy",
            "service": "Health Check Server",
            "version": "1.0.0"
        }
        self._send_json_response(200, response)
    
    def _send_json_response(self, status_code: int, data: dict):
        """JSON 응답 전송"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self._set_cors_headers()
        self.end_headers()
        
        response_json = json.dumps(data, ensure_ascii=False)
        self.wfile.write(response_json.encode('utf-8'))
    
    def _send_error(self, status_code: int, message: str):
        """에러 응답 전송"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self._set_cors_headers()
        self.end_headers()
        
        error_response = {"error": message, "status_code": status_code}
        response_json = json.dumps(error_response, ensure_ascii=False)
        self.wfile.write(response_json.encode('utf-8'))
    
    def log_message(self, format, *args):
        """로그 메시지 포맷 커스터마이징"""
        print(f"[HTTP] {format % args}")


def main():
    """서버 시작"""
    # 포트 설정 (환경변수 PORT 또는 기본값 7860)
    port = int(os.getenv('PORT', '7860'))
    host = os.getenv('HOST', '0.0.0.0')
    
    server_address = (host, port)
    httpd = HTTPServer(server_address, HealthCheckHandler)
    
    print(f"\n{'='*60}")
    print("Health Check Server Started")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Health Check: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/health")
    print(f"{'='*60}\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down server...")
        httpd.shutdown()
    except Exception as e:
        print(f"[SERVER] Server error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

