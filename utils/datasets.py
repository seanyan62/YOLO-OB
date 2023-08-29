from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
from random import sample, shuffle
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def rand(a=0.0, b=1.0):
    return np.random.rand() * (b - a) + a


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None, is_train=False):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "new_labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform
        self.train = is_train

    def __getitem__(self, index):
        if self.train:
            h, w = [416, 416]
            min_offset_x = rand(0.3, 0.7)
            min_offset_y = rand(0.3, 0.7)
            image = Image.new('RGB', (w, h), (128, 128, 128))
            indexes = sample(range(len(self.img_files)), 3)
            indexes.append(index)
            shuffle(indexes)
            boxes = []
            count = 0
            for i in indexes:
                # ---------
                #  Image
                # ---------
                try:
                    img_path = self.img_files[i % len(self.img_files)].rstrip()
                    img = Image.open(img_path)
                except Exception:
                    print(f"Could not read image '{img_path}'.")
                    return

                # ---------
                #  Label
                # ---------
                try:
                    label_path = self.label_files[i % len(self.img_files)].rstrip()
                    single_boxes = np.loadtxt(label_path).reshape(-1, 5)
                except Exception:
                    print(f"Could not read image '{label_path}'.")
                    return

                iw, ih = img.size
                if count == 0:
                    w, h = img.size
                    image = image.resize((w, h))
                    nw = int(w * min_offset_x)
                    nh = int(h * min_offset_y)
                    for box in single_boxes:
                        box[1] = box[1] * min_offset_x
                        box[2] = box[2] * min_offset_y
                        box[3] = box[3] * min_offset_x
                        box[4] = box[4] * min_offset_y
                    boxes = single_boxes
                    img = img.resize((nw, nh))
                    image.paste(img, (0, 0))
                elif count == 1:
                    nw = w - int(w * min_offset_x)
                    nh = int(h * min_offset_y)
                    for box in single_boxes:
                        box[1] = box[1] * (1 - min_offset_x) + min_offset_x
                        box[2] = box[2] * min_offset_y
                        box[3] = box[3] * (1 - min_offset_x)
                        box[4] = box[4] * min_offset_y
                    boxes = np.append(boxes, single_boxes, axis=0)
                    img = img.resize((nw, nh))
                    image.paste(img, (int(w * min_offset_x), 0))
                elif count == 2:
                    nw = int(w * min_offset_x)
                    nh = h - int(h * min_offset_y)
                    for box in single_boxes:
                        box[1] = box[1] * min_offset_x
                        box[2] = box[2] * (1 - min_offset_y) + min_offset_y
                        box[3] = box[3] * min_offset_x
                        box[4] = box[4] * (1 - min_offset_y)
                    boxes = np.append(boxes, single_boxes, axis=0)
                    img = img.resize((nw, nh))
                    image.paste(img, (0, int(h * min_offset_y)))
                elif count == 3:
                    nw = w - int(w * min_offset_x)
                    nh = h - int(h * min_offset_y)
                    for box in single_boxes:
                        box[1] = box[1]*(1-min_offset_x)+min_offset_x
                        box[2] = box[2]*(1-min_offset_y)+min_offset_y
                        box[3] = box[3]*(1-min_offset_x)
                        box[4] = box[4]*(1-min_offset_y)
                    boxes = np.append(boxes, single_boxes, axis=0)
                    img = img.resize((nw, nh))
                    image.paste(img, (int(w * min_offset_x), int(h * min_offset_y)))
                count += 1
        else:
            try:
                img_path = self.img_files[index % len(self.img_files)].rstrip()
                image = Image.open(img_path)
            except Exception:
                print(f"Could not read image '{img_path}'.")
                return
            try:
                label_path = self.label_files[index % len(self.img_files)].rstrip()
                boxes = np.loadtxt(label_path).reshape(-1, 5)
            except Exception:
                print(f"Could not read image '{label_path}'.")
                return

        image = np.array(image, dtype=np.uint8)
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # ---------
        #  Label
        # ---------
        # try:
        #     label_path = self.label_files[index % len(self.img_files)].rstrip()
        #
        #     # Ignore warning if file is empty
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore")
        #         boxes = np.loadtxt(label_path).reshape(-1, 5)
        #         # 保存拼接后的标签
        #         # concat_boxes = []
        #         # for box in boxes:
        #         #     c = box[0]
        #         #     x = box[1]/2
        #         #     y = box[2]/2
        #         #     w = box[3]/2
        #         #     h = box[4]/2
        #         #     concat_boxes.append([c, x, y, w, h])
        #         #     concat_boxes.append([c, x+0.5, y, w, h])
        #         #     concat_boxes.append([c, x, y+0.5, w, h])
        #         #     concat_boxes.append([c, x+0.5, y+0.5, w, h])
        #
        # except Exception:
        #     print(f"Could not read label '{label_path}'.")
        #     return



        plt_image = image

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                image, bb_targets = self.transform((image, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        # print(image.shape)
        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(plt_image)
        # # ax.imshow(image.permute(1, 2, 0))
        # currentAxis = fig.gca()
        # # 分别是左上的x,y, width, height
        # # rect = patches.Rectangle((bb_targets[0, [1, 2]]), bb_targets[0, 3], bb_targets[0, 4])
        # rect = patches.Rectangle((boxes[0, 1]-100, boxes[0, 2]-100), boxes[0, 3], boxes[0, 4],
        #                          linewidth=1, edgecolor='r',facecolor='none')
        #
        # currentAxis.add_patch(rect)
        # plt.show()

        return img_path, image, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)
