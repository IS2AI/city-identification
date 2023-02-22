import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import shutil
import random
import cv2
import os



def make_dir(dirName):
    # Create a target directory & all intermediate 
    # directories if they don't exists

    if not os.path.exists(dirName):
        os.makedirs(dirName, exist_ok = True)
        print("[INFO] Directory " ,dirName,  " created")
    else:
        print("[INFO] Directory " ,dirName,  " already exists")


def split_image(image_path, save_dir):
    image_name = image_path.split('/')[-1].split('.')[0]

    image = cv2.imread(image_path)
    H, W, _ = image.shape

    cropped_half1 = image[0:H, 0:round(W/2)]
    cropped_half2 = image[0:H, round(W/2):W]

    make_dir(os.path.join(save_dir, image_name))
    print("path for saving splitted images:" + os.path.join(save_dir, image_name))
    save_path = os.path.join(save_dir, image_name, image_name + '_half1.jpg')
    print("cropped half1: " + save_path)
    cv2.imwrite(save_path, cropped_half1)
    save_path = os.path.join(save_dir, image_name, image_name + '_half2.jpg')
    print("cropped half2: " + save_path)
    cv2.imwrite(save_path, cropped_half2)



def make_patches(image_path, patch_dim, stride, save_dir):
    print('image path: '+ image_path)
    image_name = image_path.split('/')[-1].split('.')[0]

    image = cv2.imread(image_path)
    H, W, _ = image.shape

    if patch_dim > H or patch_dim > W:
        raise Exception("Patch size exceeds image dimensions.")
    
    make_dir(os.path.join(save_dir, image_name))

    N = 0
    i, j = 0, 0
    while i + patch_dim <= H:
        while j + patch_dim <= W:

            cropped_image = image[i:i + patch_dim, j:j + patch_dim, :]

            save_path = os.path.join(save_dir, image_name, image_name + '_{}.jpg'.format(N))
            print("patched image: " + save_path)
            cv2.imwrite(save_path, cropped_image)

            print(save_path, i, i+patch_dim, j, j+patch_dim)
            N += 1
            j += stride
        j = 0
        i += stride

SET = 'patches'
ROOT = 'preprocessed'
CITIES = os.listdir(os.path.join(ROOT, 'raw'))
CITIES.remove('.ipynb_checkpoints')

city_codes = []
CITIES.sort()
for CITY in CITIES:
    city_code = CITY.split('(')[-1].split(')')[0]
    city_codes.append(city_code)
print(city_codes)
print(CITIES)

for CITY in CITIES:
    print(CITY)

    # get sites
    SITES = os.listdir(os.path.join(ROOT, 'raw', CITY))
    SITES.sort()
    print(SITES)
    print(SITES[len(SITES)-1])
    # for each site, make patches. For the last site split the image
    for SITE in SITES:
        print(SITE)

        image_paths = list(paths.list_files(os.path.join(ROOT, 'raw', CITY, SITE), validExts='jpg'))
        save_dir = os.path.join(ROOT, SET, CITY, SITE)
        make_dir(save_dir)

        for image_path in image_paths:
            if SITE == SITES[len(SITES)-1]:
                split_image(image_path, save_dir) # split images in the last folder
            else:
                make_patches(image_path, 480, 240, save_dir)

    # create patches for the last site
    LAST = SITES[-1]
    
    image_paths = list(paths.list_files(os.path.join(ROOT, SET, CITY, LAST), validExts='jpg'))
    save_dir = os.path.join(ROOT, SET, CITY, LAST)
    
    for image_path in image_paths:
        print("making patches for " + image_path)
        make_patches(image_path, 480, 240, save_dir)
        os.remove(image_path)

    # use first two sites as training set
    make_dir(os.path.join(ROOT, SET, 'train', CITY))

    for SITE in SITES[:-1]:
        image_paths = list(paths.list_files(os.path.join(ROOT, SET, CITY, SITE), validExts='jpg'))

        for image_path in image_paths:
            image_name = image_path.split('/')[-1]
            os.rename(image_path, os.path.join(ROOT, SET, 'train', CITY, image_name))

    # use last site as validation and test sets
    make_dir(os.path.join(ROOT, SET, 'test', CITY))
    make_dir(os.path.join(ROOT, SET, 'val', CITY))


    image_paths = list(paths.list_files(os.path.join(ROOT, SET, CITY, LAST), validExts='jpg'))
    for image_path in image_paths:
            image_name = image_path.split('/')[-1]
            
            if image_name.find('half1_') == -1:
                os.rename(image_path, os.path.join(ROOT, SET, 'val', CITY, image_name))
            elif image_name.find('half2_') == -1:
                os.rename(image_path, os.path.join(ROOT, SET, 'test', CITY, image_name))

for CITY in CITIES:
    make_dir(os.path.join(ROOT, SET, 'test_b', CITY))

    image_paths = list(paths.list_files(os.path.join(ROOT, SET, 'test', CITY), validExts='jpg'))
    selected_paths = random.sample(image_paths, 500)

    for path in selected_paths:
        new_path = path.replace('test', 'test_b')
        shutil.copyfile(path, new_path)


# statistics
for CITY in CITIES:
    train_paths = list(paths.list_files(os.path.join(ROOT, SET, 'train', CITY), validExts='jpg'))
    val_paths = list(paths.list_files(os.path.join(ROOT, SET, 'val', CITY), validExts='jpg'))
    test_paths = list(paths.list_files(os.path.join(ROOT, SET, 'test', CITY), validExts='jpg'))

    print("{}: {} {} {}".format(CITY, len(train_paths), len(val_paths), len(test_paths)))
