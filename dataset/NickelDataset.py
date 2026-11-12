# Основной класс для подготовки датасета  и нарезки на патчи
import numpy as np
from matplotlib import pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class NikelDataset(Dataset):
    def __init__(self, images, segmentation_masks, patch_height):

        self.images = images
        self.segmentation_masks = segmentation_masks
        self.patch_height = patch_height

        # Считаем высоту каждого изображения в папке, чтобы рассчитать, сколько полос 2700хpatch_height
        # получится из каждого изображения.
        # После этого создадим массив с максимальным индеком для каждого изображения,
        # чтобы обращаться к нему в методе __getitem__
        self.total_patches_amount = 0
        self.index_intervals = [0]
        for image_file in images:
            # Open the image file
            with Image.open(image_file) as img:
                # Get width and height as a tuple
                width, height = img.size
                # print(f'Image {os.path.basename(image_file)} WH: {width} {height}')

                # Расчитываем, сколько патчей получится из одного изобраежния и обновляем общее количество патчей
                cur_img_patches_num = int(height / patch_height) # Loose bottom
                # idx_start = self.total_patches_amount
                idx_end = self.total_patches_amount + cur_img_patches_num
                self.total_patches_amount += cur_img_patches_num
                self.index_intervals.append(idx_end)
        self.index_intervals = np.array(self.index_intervals)



        self.images_transforms = transforms.Compose([
            # transforms.Resize((256, 256)), # размер каждой картинки будет приведен к 256*256
            transforms.ToTensor(), # картинка из питоновского массива переводится в формат torch.Tensor
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # значения пикселей картинки нормализуются
        ])

        self.masks_transforms = transforms.Compose([
            # используем InterpolationMode.NEAREST, чтобы при изменении размера
            # маски сегментации не менялись номера классов
            # transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST), # размер каждой картинки будет приведен к 256*256
            transforms.ToTensor(), # картинка из питоновского массива переводится в формат torch.Tensor
        ])


    def get_patch_by_idx_(self, index):
        # Расчитываем индекс и режем соответствующий патч
        # 1 вычитаем потому что у нас в массиве интервалов первый 0
        img_index = len(self.index_intervals[self.index_intervals <= index])-1
        patch_index = index - self.index_intervals[self.index_intervals <= index][-1]
        # Считаем, как обрезать изображение (left, upper, right, lower)
        img = Image.open(self.images[img_index])
        width = img.size[0]
        crop_box = (0,
                    self.patch_height * patch_index,
                    width,
                    self.patch_height * (patch_index + 1))
        return img.crop(crop_box)

    def get_hard_mask_by_idx_(self, index, mask_priority=(0,1,2,3)):
        # Расчитываем индекс и режем соответствующую маску
        # 1 вычитаем потому что у нас в массиве интервалов первый 0
        # Считав все маски, схлопываем их в одну многоканальную
        # Если пиксель попадает на два класса, берется более приоритетный класс из mask_priority
        mask_index = len(self.index_intervals[self.index_intervals <= index])-1
        patch_index = index - self.index_intervals[self.index_intervals <= index][-1]
        # Считаем, как обрезать изображение (left, upper, right, lower)
        
        mask = Image.open(self.segmentation_masks[mask_index])
        width = mask.size[0]
        crop_box = (0,
                    self.patch_height * patch_index,
                    width,
                    self.patch_height * (patch_index + 1))
        return mask.crop(crop_box)
            

    def __getitem__(self, index):
        '''
        этот метод должен по заданному номеру пары картинка-сегментация (index)
        возвращать эту пару. Этот метод обязательно нужно реализовывать во всех
        кастомных классах Dataset. Перед тем, как выдать на выходе
        пару картинка-сегментация, можно применить к картинкам любые преобразования:
        например, знакомую нам аугментацию.
        '''

        # загружаем нужные картинку и ее карту сегментации
        img = Image.open(self.images[index])
        mask = Image.open(self.segmentation_masks[index])

        # # применяем предобработку к RGB картинке
        img = self.images_transforms(img)
        # # применяем предобработку к маске сегментации
        mask = self.masks_transforms(mask)*255

        # # делим маску сегментации на 13 бинарных масок сегментации
        # # отдельно для каждого класса
        masks = []

        # вытаскиваем пиксели, принадлежащие классам .
        for i in [1, 2, 3, 4]:
            # генерируем бинарную маску сегментации для текущего класса i
            cls_mask = torch.where(mask==i, 1, 0)
            masks.append(cls_mask[0,:,:].numpy())

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # схлопываем бинарные карты сегментации в одну
        masks = torch.argmax(masks, axis=0)

        # возвращаем пару: картинка — ее маска сегментации на 4 класса
        return (img, masks)

    def __len__(self):
        '''
        этот метод должен возвращать количество пар картинка-сегментация в датасете
        '''
        return self.total_patches_amount