# use this import if running train.py in this dir
from . import transforms as T

# use this import if running test_MSD.py
# from . import transforms as T


class DetectionPresetTrain:
    def __init__(self, hflip_prob=0.5):
        trans = [T.ToTensor()]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))

        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        self.transforms = T.ToTensor()

    def __call__(self, img, target):
        return self.transforms(img, target)
