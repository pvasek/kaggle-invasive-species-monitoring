from torchvision import datasets

class ImageFolder2(datasets.ImageFolder):
    def __getitem__(self, index):
        original_item = super(ImageFolder2, self).__getitem__(index)
        path, _ = self.imgs[index]
        return original_item, path
