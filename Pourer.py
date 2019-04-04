import cv2 as cv
import os
from DataHolder import DataHolder
from PIL import Image
from Classificator import Classificator
from RectangleDetector import RectangleDetector

source_images_folder = "resources\\images\\source_images"

files = [os.path.join(source_images_folder, p) for p in sorted(os.listdir(source_images_folder))]
result = {}
#kls = Classificator.from_folder()
kls = Classificator.from_pickle()
for file in files:
    dict_file_names = file.split('\\')
    file_name = dict_file_names[len(dict_file_names) - 1]
    img = cv.imread(file)
    rd = RectangleDetector(img)
    cropped_images = rd.crop()
    if cropped_images:
        entities = []
        i = 0
        for cropped_image in cropped_images:
            i += 1
            # cv.imshow("cr", cropped_image)
            # cv.waitKey(0)
            loc = '{}\\{}-{}'.format("resources\\images\\cropped_images", i, file_name)
            cv.imwrite(loc, cropped_image)
            dh = DataHolder(file_name)
            dh.set_cropped_image(cropped_image)
            entities.append(dh)
        i = 0
        for entity in entities:
            i += 1
            cv_image = entity.cropped_image
            pil_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(pil_image)
            closest = kls.classify(pil_image)
            entity.recognised_name = closest
            loc = '{}\\{}-{}'.format("resources\\images\\recognised_images", i, closest)
            cv.imwrite(loc, cv_image)
            print("-------------")
            print(file_name, closest)
            print("-------------")

        cv.destroyAllWindows()
        #print("end")

