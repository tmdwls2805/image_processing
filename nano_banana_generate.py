from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os
import json
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# 이미지 로드 (다른 이미지로 교체 가능)
model_image = Image.open('file/trump.jpg')
dress_image = Image.open('file/dress2.jpg')

# JSON 기반 프롬프트 (범용적 설명)
prompt_data = {
    "person": {
        "description": (
            "첨부된 이미지들 중에서 한 사람만 사용하세요: "
            "사진에서 가장 큰 면적을 차지하는 단일 인물을 선택하세요. "
            "이 사람이 합성의 주요 피사체가 됩니다."
        ),
        "reference_image": f"{model_image}"
    },
    "clothing": {
        "description": (
            "의상 참조 이미지에서 옷 자체만 가이드로 사용하세요. "
            "입고 있는 모델은 포함하지 말고, 의상 디자인만 추출하세요. "
            "이 옷은 사람 이미지에서 선택된 인물에게 입혀져야 합니다."
        ),
        "reference_image": f"{dress_image}"
    },
    "background": {
        "description": (
            "배경 참조 설명를 환경으로 사용하세요. "
            "단순히 사람을 위에 붙여넣지 말고, 자연스럽게 배경에 녹여내어 "
            "최종 결과가 매끄럽고 사실적으로 보이도록 하세요. "
            "마치 함께 촬영된 것처럼 보여야 합니다. 결과물은 "
            "다른 사람들이 보기에도 진짜처럼 보일 정도로 설득력이 있어야 합니다."
        ),
        "reference_text": "깊은 바닷속"
    },
    "style": {
        "lighting": "부드럽고 자연스러운 조명",
        "tone": "패션 에디토리얼, 높은 디테일, 사실적인 옷감 질감",
        "camera": "인물 샷, 전신 뷰",
        "pose": "카메라를 보며 미소 짓고 행복한 사람처럼 점프"
    }
}


# JSON → 문자열
prompt = json.dumps(prompt_data, indent=2)

# 이미지 + 프롬프트 결합
response = client.models.generate_content(
    model="gemini-2.5-flash-image-preview",
    contents=[
        model_image,        # 사람
        dress_image,        # 옷
        prompt              # JSON 설명
    ],
)

# 응답에서 이미지 추출
image_parts = [
    part.inline_data.data
    for part in response.candidates[0].content.parts
    if part.inline_data
]

# 저장 및 보기
if image_parts:
    image = Image.open(BytesIO(image_parts[0]))
    image.save('file/generated_image2.png')
    image.show()
