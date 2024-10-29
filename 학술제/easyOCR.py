from easyocr import Reader
import cv2

class myOCR:
    def __init__(self, langs=['ko', 'en']):
            myOCR.reader = Reader(lang_list=langs, gpu=False)

    def predict(self, image):
        preprocessed_image = cv2.imread(image)
        results = myOCR.reader.readtext(preprocessed_image)
        
        extracted_text = ""
        for (bbox, text, prob) in results:
            print(f"[INFO] {prob:.4f}: {text}")
            extracted_text += text + "\n"
        
        return extracted_text

# 사용
image = '/Users/jy_tony3/sjy/2024/학교/프로젝트/학술제/OCR 데이터셋/Training/01.원천데이터/test3.png'
ocr_model = myOCR()
extracted_text = ocr_model.predict(image)
print("\n추출된 텍스트:\n", extracted_text)
