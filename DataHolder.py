class DataHolder:
    def __init__(self, original_image_file_name):
        self.org_img_file_name = original_image_file_name
        self.cropped_image = None
        self.recognised_name = None

    def __repr__(self):
        if self.recognised_name is not None:
            return 'DataHolder("{}", "{}")'.format(self.org_img_file_name, self.recognised_name)
        return 'DataHolder("{}")'.format(self.org_img_file_name)

    def set_cropped_image(self, cropped_image):
        self.cropped_image = cropped_image
