import os
import logging
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)

# Gemini API 클라이언트 초기화
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None
    logger.warning("GEMINI_API_KEY is not set")

class GeminiAgeTransformer:
    def __init__(self, api_key=None):
        """Gemini API를 사용한 나이 변환 클래스"""
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash-image"

    def load_image_as_base64(self, image_path):
        """로컬 이미지 파일을 base64로 로드"""
        try:
            print(f"[Load] 이미지 로드 중: {image_path}")

            # 이미지 파일을 바이너리로 읽기
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # MIME type 결정
            if image_path.lower().endswith('.png'):
                mime_type = 'image/png'
            elif image_path.lower().endswith(('.jpg', '.jpeg')):
                mime_type = 'image/jpeg'
            elif image_path.lower().endswith('.webp'):
                mime_type = 'image/webp'
            else:
                mime_type = 'image/jpeg'  # 기본값

            print(f"[Load] 로드 완료 (MIME: {mime_type})")
            return image_data, mime_type

        except Exception as e:
            print(f"[Error] 이미지 로드 실패: {e}")
            logger.error(f"Image load failed: {e}")
            return None, None

    def _get_fixed_requirements(self):
        """나이 변환 시 항상 유지해야 하는 고정적인 요구사항"""
        return """
        ⚠️ 중요: 반드시 원본 인물의 얼굴 형태, 눈 모양, 코 형태, 입술 모양, 얼굴 윤곽을 정확히 유지해야 합니다.
        다른 사람처럼 보이면 안 됩니다. 동일 인물이 나이만 든 것처럼 보여야 합니다.

        고정 요구사항 (절대 변경 금지):
        1. ✅ 얼굴 구조(뼈대), 눈·코·입 위치와 크기, 얼굴형, 눈썹 모양을 원본과 100% 동일하게 유지
        2. ✅ 표정과 시선 방향도 원본과 동일하게 유지
        3. ✅ 배경은 원본과 완전히 동일하게 유지
        """

    def _get_aging_effects(self, intensity="medium"):
        """
        나이에 따라 변하는 가변적인 효과

        Args:
            intensity: "light" (약간), "medium" (중간), "heavy" (강함)
        """
        effects = {
            "light": """
            가변 요구사항 (나이 효과 - 가볍게):
            - 눈가와 입가에 약간의 잔주름 추가 (fine lines)
            - 피부에 약간의 질감 추가
            - 머리카락에 약간의 흰머리 추가 (10-20%)
            """,
            "medium": """
            가변 요구사항 (나이 효과 - 중간):
            - 이마, 눈가, 입가에 주름 추가 (crow's feet, forehead lines, nasolabial folds)
            - 피부 탄력을 약간 줄이고 처진 느낌 추가 (slight sagging)
            - 피부 톤을 조금 어둡고 칙칙하게 (age spots, uneven skin tone)
            - 눈밑에 다크서클과 약간의 눈꺼풀 처짐 추가
            - 머리카락에 흰머리 추가 (30-50%)
            - 목에 약간의 주름 추가
            - 피부 질감을 약간 거칠게 만들기
            """,
            "heavy": """
            가변 요구사항 (나이 효과 - 강하게):
            - 이마, 눈가, 입가에 깊은 주름 추가 (deep crow's feet, forehead lines, nasolabial folds)
            - 피부 탄력을 크게 줄이고 처진 느낌 추가 (sagging skin, jowls)
            - 피부 톤을 더 어둡고 칙칙하게 (prominent age spots, uneven skin tone)
            - 눈밑에 두드러진 다크서클과 눈꺼풀 처짐 추가
            - 머리카락에 많은 흰머리 추가하거나 머리숱 크게 감소 (60-80% gray/white hair, hair thinning)
            - 목 주름과 목 처짐 추가 (neck wrinkles, turkey neck)
            - 피부 질감을 거칠고 윤기 없게 만들기
            - 피부에 검버섯이나 잡티 추가
            """
        }
        return effects.get(intensity, effects["medium"])

    def transform_age(self, image_path, target_age_description="20살처럼 젊고 동안으로", aging_intensity="medium"):
        """
        이미지의 얼굴을 지정된 나이로 변환

        Args:
            image_path: 입력 이미지 경로
            target_age_description: 목표 나이 설명 (예: "20살처럼 젊고 동안으로", "50대처럼 성숙하고 나이 들어 보이게")
            aging_intensity: 나이 효과 강도 ("light", "medium", "heavy")

        Returns:
            생성된 이미지의 base64 데이터 또는 None
        """
        try:
            # 이미지 로드
            image_data, mime_type = self.load_image_as_base64(image_path)
            if not image_data:
                return None

            # 프롬프트 생성: 고정 부분 + 가변 부분
            fixed_reqs = self._get_fixed_requirements()
            aging_effects = self._get_aging_effects(aging_intensity)

            prompt = f"""이 사진 속 인물의 얼굴을 {target_age_description} 변환해주세요.

            {fixed_reqs}

            {aging_effects}

            자연스럽지만 명확하게 나이 들어 보이는 이미지를 생성해주세요.
            단, 얼굴의 핵심 특징(identity)은 절대 변경하지 마세요."""

            print(f"[Generate] 이미지 생성 중...")
            print(f"[Prompt] {prompt}")

            # 이미지 생성 요청 (인라인 데이터 사용)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(
                                data=image_data,
                                mime_type=mime_type
                            ),
                            types.Part.from_text(text=prompt)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    top_p=0.95,
                    top_k=20,
                    max_output_tokens=8192,
                    response_modalities=["IMAGE"]
                )
            )

            print(f"[Response] 응답 받음")

            # 응답 처리
            if response and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        print(f"[Success] 이미지 생성 완료")
                        return part.inline_data.data

            print(f"[Warning] 생성된 이미지를 찾을 수 없습니다")
            print(f"[Debug] Response: {response}")
            return None

        except Exception as e:
            print(f"[Error] 이미지 변환 실패: {e}")
            logger.error(f"Age transformation failed: {e}")
            return None

    def save_image(self, image_data, output_path):
        """
        base64 이미지 데이터를 파일로 저장

        Args:
            image_data: base64 인코딩된 이미지 데이터
            output_path: 저장할 파일 경로
        """
        try:
            print(f"[Save] 이미지 저장 중: {output_path}")

            # base64 디코드
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data

            # PIL Image로 변환
            img = Image.open(BytesIO(image_bytes))

            # 파일 확장자에 따라 저장
            if output_path.endswith('.webp'):
                img.save(output_path, 'WEBP', quality=90)
            elif output_path.endswith('.jpg') or output_path.endswith('.jpeg'):
                img.save(output_path, 'JPEG', quality=90)
            elif output_path.endswith('.png'):
                img.save(output_path, 'PNG')
            else:
                # 기본값: WebP
                img.save(output_path, 'WEBP', quality=90)

            print(f"[Saved] 이미지 저장 완료: {output_path}")
            return True

        except Exception as e:
            print(f"[Error] 이미지 저장 실패: {e}")
            logger.error(f"Image save failed: {e}")
            return False


if __name__ == "__main__":
    # 환경변수에서 API 키 가져오기
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("[Error] GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("사용법: export GEMINI_API_KEY='your-api-key'")
        exit(1)

    # GeminiAgeTransformer 인스턴스 생성
    transformer = GeminiAgeTransformer(api_key=api_key)

    # 입력 이미지 경로 설정
    input_image_path = "man_face.jpeg"

    # 이미지 파일 존재 확인
    if not os.path.exists(input_image_path):
        print(f"[Error] 입력 이미지를 찾을 수 없습니다: {input_image_path}")
        print("사용법: 이미지 파일을 준비하고 input_image_path 변수를 수정하세요")
        exit(1)

    print("=" * 50)
    print("Gemini 얼굴 나이 변환 시작")
    print("=" * 50)

    # 나이 변환 실행
    target_age_description = "50대처럼 성숙하고 나이 들어 보이게"
    image_data = transformer.transform_age(input_image_path, target_age_description)

    if image_data:
        # result 디렉토리 생성
        os.makedirs("result", exist_ok=True)

        # 결과 저장
        output_path = "result/gemini_age_transformed.webp"
        success = transformer.save_image(image_data, output_path)

        if success:
            print("\n" + "=" * 50)
            print(f"변환 완료! 결과 이미지: {output_path}")
            print("=" * 50)
        else:
            print("\n[Error] 이미지 저장에 실패했습니다")
    else:
        print("\n[Error] 이미지 변환에 실패했습니다")