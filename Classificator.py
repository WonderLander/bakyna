import imagehash
import os
import pickle
from PIL import Image


class Classificator:

    def __init__(self, dict_hashed):
        self.dist_hashes = dict_hashed

    @classmethod
    def from_pickle(cls, pickle_file_name="resources\\pickles\\hashes.pck"):
        with open(pickle_file_name, 'rb') as handle:
            dh = pickle.load(handle)
        return cls(dh)

    @classmethod
    def from_folder(cls, folder_name="resources\\images\\images",
                    do_pickle=True, pickle_file_name="resources\\pickles\\hashes.pck"):
        dict_hashed = cls.__batch_hash(folder_name)
        if do_pickle:
            with open(pickle_file_name, 'wb') as handle:
                pickle.dump(dict_hashed, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return cls(dict_hashed)

    @staticmethod
    def __hash(img):
        img_hash = imagehash.dhash(img, 16)
        #img_hash = imagehash.phash(img)
        return img_hash

    @classmethod
    def __batch_hash(cls, images_folder):
        files = [os.path.join(images_folder, p) for p in sorted(os.listdir(images_folder))]
        result = {}

        for file in files:
            print('hashing image %s' % file)
            dict_file_names = file.split('\\')
            file_name = dict_file_names[len(dict_file_names) - 1]
            img = Image.open(file)
            result[file_name] = cls.__hash(img)
        return result

    def classify(self, img):
        img_hash = self.__hash(img)
        shortest_dist = 1100
        closest = None
        for key, value in self.dist_hashes.items():
            dist = img_hash - value
            if shortest_dist > dist:
                shortest_dist = dist
                closest = key
        print(dist)
        return closest



