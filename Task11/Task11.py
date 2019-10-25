# -*- coding: UTF-8 -*-

from PIL import Image
import pytesseract

print(pytesseract.image_to_string(Image.open('test.jpg'), lang='eng'))