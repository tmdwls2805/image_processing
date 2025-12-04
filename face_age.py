import requests
import base64
import time
import json
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()   

class WondershareAgeAPI:
    def __init__(self, app_key, app_secret):
        # 공식 문서 및 테스트를 통해 확인된 호스트
        self.base_url = "https://wsai-api.wondershare.com"
        self.app_key = app_key
        self.app_secret = app_secret
        self.headers = self._get_auth_headers()

    def _get_auth_headers(self):
        """
        app_key와 app_secret을 사용하여 Basic Auth 헤더를 생성합니다.
        Format: Basic base64(appkey:appsecret)
        """
        credentials = f"{self.app_key}:{self.app_secret}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        return {
            "Content-Type": "application/json",
            "Authorization": f"Basic {encoded_credentials}"
        }

    def create_task(self, image_urls, target_age):
        """
        1단계: 나이 변환 작업 생성 (POST)
        - priority 파라미터는 에러 유발 가능성이 있어 제거했습니다.
        """
        endpoint = "/v3/pic/at/batch"
        url = f"{self.base_url}{endpoint}"
        
        # [수정됨] priority 제거 및 age 정수 변환 강제
        payload = {
            "images": image_urls,
            "age": int(target_age)
        }

        print(f"[Request] URL: {url}")
        print(f"[Request] Payload: {json.dumps(payload, ensure_ascii=False)}") # 전송 데이터 확인

        try:
            response = requests.post(url, headers=self.headers, json=payload)
            
            # 에러 발생 시 상세 내용 출력을 위한 로직
            if response.status_code != 200:
                print(f"[Error] Status Code: {response.status_code}")
                print(f"[Error] Response Content: {response.text}")
            
            response.raise_for_status() 
            result = response.json()
            
            if result.get("code") == 0:
                task_id = result["data"]["task_id"]
                print(f"[Task Created] ID: {task_id}")
                return task_id
            else:
                print(f"[Error] Task Creation Failed: {result.get('msg')} (Code: {result.get('code')})")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"[Error] Request failed: {e}")
            return None

    def get_result(self, task_id, max_retries=3, interval=3):
        """
        2단계: 결과 조회 및 폴링 (GET)
        """
        endpoint = f"/v3/pic/at/result/{task_id}"
        url = f"{self.base_url}{endpoint}"

        print(f"[Polling] URL: {url}")

        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                
                task_data = data.get("data", {})
                status = task_data.get("status") 
                # status: 1(대기), 2(진행중), 3(완료), 4(실패), 6(타임아웃)
                
                if status == 3:
                    print("[Success] 작업 완료!")
                    return task_data.get("list", [])
                
                elif status == 4:
                    print(f"[Failed] 작업 실패 사유: {task_data.get('reason')}")
                    return None
                    
                elif status in [1, 2]:
                    # API가 제안하는 대기 시간이 있으면 사용, 없으면 기본값 사용
                    wait_time = float(task_data.get("wait_time", interval))
                    if wait_time < 1: wait_time = interval
                    
                    print(f"[{attempt+1}/{max_retries}] 처리 중... (상태: {status}), {wait_time}초 대기")
                    time.sleep(wait_time)
                    
                else:
                    print(f"[Error] 알 수 없는 상태 코드: {status}, 응답: {data}")
                    time.sleep(interval)
                    
            except requests.exceptions.RequestException as e:
                print(f"[Error] Polling failed: {e}")
                time.sleep(interval)
        
        print("[Timeout] 최대 재시도 횟수 초과")
        return None

if __name__ == "__main__":
    # 1. 환경변수에서 키 가져오기
    APP_KEY = os.getenv("AI_LAB_APP_KEY")       
    APP_SECRET = os.getenv("AI_LAB_APP_SECRET") 
    
    if not APP_KEY or not APP_SECRET:
        print("[Error] .env 파일에 AI_LAB_APP_KEY 또는 AI_LAB_APP_SECRET이 없습니다.")
        exit()

    # 2. 테스트할 이미지 URL 설정
    # 주의: "hello2.jpg" 같은 로컬 파일 경로는 사용할 수 없습니다. 반드시 http로 시작하는 웹 URL이어야 합니다.
    # 아래는 Wondershare 공식 테스트용 이미지입니다.
    IMAGE_URLS = ["https://media.helloeveryoung.com/test/hello2.jpg"]
    
    # 3. 목표 나이 설정
    TARGET_AGE = 60
    
    api = WondershareAgeAPI(APP_KEY, APP_SECRET)
    
    print("--- 1. 작업 생성 요청 ---")
    # priority 파라미터는 함수 내부에서 제거되었으므로 인자로 넘기지 않습니다.
    task_id = api.create_task(IMAGE_URLS, TARGET_AGE)
    
    if task_id:
        print("\n--- 2. 결과 조회 요청 ---")
        results = api.get_result(task_id)
        
        if results:
            print("\n--- 결과 ---")
            for item in results:
                print(f"원본 이미지: {item.get('image_url')}")
                print(f"변환된 이미지: {item.get('image_result')}")
