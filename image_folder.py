from torchvision import datasets

class ImageFolder2(datasets.ImageFolder):
    def __getitem__(self, index):
        original_item = super(ImageFolder2, self).__getitem__(index)
        path, _ = self.imgs[index]
        # print(f'ImageFolder2 path: {path}')
        return original_item, path
