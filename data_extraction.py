import os
from math import cos, radians
from constants import *
import sys

import cv2
from PIL import Image
import pickle

import numpy as np
import logging


log = logging.getLogger("my-logger")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

FDDB_FOLD_ELLIPSE_FILES = [i for i in os.listdir(DIR_PATH + FDDB_FILE_PATH) if 'ellipse' in i]

def convert_ellipse_to_bbox():
    os.makedirs(DIR_PATH + OUTPUT_DIR, exist_ok=True)
    os.makedirs(DIR_PATH + OUTPUT_DIR + BBOX_FDDB_FOLDS, exist_ok=True)
    for ellipse_file in FDDB_FOLD_ELLIPSE_FILES:
        log.info('****************Parsing {}**********************'.format(ellipse_file))
        with open(DIR_PATH + FDDB_FILE_PATH + ellipse_file) as f:
            lines = [line.rstrip('\n') for line in f]
        line_num = 0
        with open(DIR_PATH + OUTPUT_DIR + BBOX_FDDB_FOLDS + ellipse_file.split('ellipse')[0] + "BoundingBox.txt",
                  "w") as bounding_box_file:
            while line_num < len(lines):
                img_path = lines[line_num]

                with Image.open(DIR_PATH + IMAGES_DIR + img_path + IMG_FORMAT) as img:
                    img_width, img_height = img.size
                num_faces = int(lines[line_num + 1])

                bounding_boxes = []
                for i in range(num_faces):
                    major_axis_radius, minor_axis_radius, angle, center_x, center_y, _ = [float(l) for l in lines[
                        line_num + 2 + i].split()]

                    rect_height = 2 * major_axis_radius * (cos(radians(abs(angle))))
                    rect_width = 2 * minor_axis_radius * (cos(radians(abs(angle))))

                    left_x = int(max(0, center_x - rect_width / 2))
                    left_y = int(max(0, center_y - rect_height / 2))
                    right_x = int(min(img_width - 1, center_x + rect_width / 2))
                    right_y = int(min(img_height - 1, center_y + rect_height / 2))

                    bounding_boxes.append([left_x, left_y, right_x, right_y])

                bounding_box_file.write(img_path + "\n")
                bounding_box_file.write(str(num_faces) + "\n")
                for bb_box in bounding_boxes:
                    bounding_box_file.write(" ".join([str(i) for i in bb_box]) + "\n")
                line_num += num_faces + 2
        log.info('****************Completed Parsing {}**********************'.format(ellipse_file))


# extract face data in the form of bbox
def extract_faces_from_images(bounding_box_files):
    for bounding_box_file in bounding_box_files:
        with open(DIR_PATH + OUTPUT_DIR + BBOX_FDDB_FOLDS + bounding_box_file) as f:
            lines = [l.strip("\n") for l in f]
        line_num = 0
        while line_num < len(lines):
            img_path = lines[line_num]

            img_dir = img_path.rpartition("/")[0]
            img_name = img_path.rpartition("/")[-1]
            os.makedirs(DIR_PATH + OUTPUT_DIR + BBOX_FACES_DIR + img_dir, exist_ok=True)
            img = cv2.imread(DIR_PATH + IMAGES_DIR + img_path + IMG_FORMAT)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float')

            num_faces = int(lines[line_num + 1])
            for i in range(num_faces):
                left_x, left_y, right_x, right_y = [int(j) for j in lines[line_num + 2 + i].split()]
                crop_img = img[left_y:right_y, left_x:right_x]
                cv2.imwrite(DIR_PATH + OUTPUT_DIR + BBOX_FACES_DIR + \
                            img_dir + "/" + img_name + "_" + str(i) + IMG_FORMAT, \
                            cv2.resize(crop_img, FACE_DIM))
            line_num += num_faces + 2


def get_files(path):
    files = []
    for root, subdirs, images in os.walk(path):
        if images:
            full_path_images = [os.path.join(root, image).replace("\\", "/") for image in images]
            files.extend(full_path_images)
    return files


def generate_non_face_image_coordinates(face_coordinates, img_width, img_height):
    w, h = img_width, img_height
    if img_width < w and img_height < h:
        return set()

    non_face_coordinates = set()

    for x in range(img_width - 20):
        for y in range(img_height - 20):
            lx, ly, rx, ry = x, y, x + w, y + h
            for llx, lly, rrx, rry in face_coordinates:
                if ((rx <= llx) or (ry <= lly) or (lx >= rrx) or (ly >= rry)):
                    non_face_coordinates.add((lx, ly, rx, ry))
                    if len(non_face_coordinates) == len(face_coordinates):
                        return non_face_coordinates
    return non_face_coordinates


# extract non face data in the form of bbox
def extract_non_faces_from_images(bounding_box_files):
    for bounding_box_file in bounding_box_files:
        with open(DIR_PATH + OUTPUT_DIR + BBOX_FDDB_FOLDS + bounding_box_file) as f:
            lines = [l.strip("\n") for l in f]
        line_num = 0
        while line_num < len(lines):
            img_path = lines[line_num]

            img_dir = img_path.rpartition("/")[0]
            img_name = img_path.rpartition("/")[-1]
            os.makedirs(DIR_PATH + OUTPUT_DIR + BBOX_NON_FACES_DIR + img_dir, exist_ok=True)
            img = cv2.imread(DIR_PATH + IMAGES_DIR + img_path + IMG_FORMAT)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float')
            img_width, img_height = FACE_DIM

            num_faces = int(lines[line_num + 1])
            face_coordinates = set()
            for i in range(num_faces):
                left_x, left_y, right_x, right_y = [int(j) for j in lines[line_num + 2 + i].split()]
                face_coordinates.add((left_x, left_y, right_x, right_y))
            non_face_coordinates = generate_non_face_image_coordinates(face_coordinates, img_width, img_height)

            for idx, j in enumerate(non_face_coordinates):
                lx, ly, rx, ry = j
                crop_img = img[ly:ry, lx:rx]
                cv2.imwrite(DIR_PATH + OUTPUT_DIR + BBOX_NON_FACES_DIR + \
                            img_dir + "/" + img_name + "_non_face_" + str(idx) + IMG_FORMAT, \
                            cv2.resize(crop_img, FACE_DIM))
            line_num += num_faces + 2


def data_preparation():
    face_files = get_files(DIR_PATH + OUTPUT_DIR + BBOX_FACES_DIR)
    non_face_files = get_files(DIR_PATH + OUTPUT_DIR + BBOX_NON_FACES_DIR)

    tr_face_data_images = face_files[:TRAIN_SAMPLES]
    tr_non_face_data_images = non_face_files[:TRAIN_SAMPLES]

    te_face_data_images = face_files[TRAIN_SAMPLES:TRAIN_SAMPLES + TEST_SAMPLES]
    te_non_face_data_images = non_face_files[TRAIN_SAMPLES:TRAIN_SAMPLES + TEST_SAMPLES]

    tr_face_data = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in tr_face_data_images]
    tr_non_face_data = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in tr_non_face_data_images]

    te_face_data = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in te_face_data_images]
    te_non_face_data = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in te_non_face_data_images]

    tr_face_data = np.array(
        [cv2.normalize(i, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for i in tr_face_data])
    tr_non_face_data = np.array(
        [cv2.normalize(i, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for i in
         tr_non_face_data])

    te_face_data = np.array(
        [cv2.normalize(i, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for i in te_face_data])
    te_non_face_data = np.array(
        [cv2.normalize(i, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for i in
         te_non_face_data])

    tr_face_labels = np.array([1] * TRAIN_SAMPLES)
    tr_non_face_labels = np.array([0] * TRAIN_SAMPLES)

    te_face_labels = np.array([1] * TEST_SAMPLES)
    te_non_face_labels = np.array([0] * TEST_SAMPLES)

    return tr_face_data, tr_non_face_data, tr_face_labels, tr_non_face_labels, \
           te_face_data, te_non_face_data, te_face_labels, te_non_face_labels


def main(normalize=True):
    log.info("Converting all ellipse shaped faces to BBox")
    convert_ellipse_to_bbox()
    log.info("Converted all ellipse shaped faces to BBox")
    bounding_box_files = [f for f in os.listdir(DIR_PATH + OUTPUT_DIR + BBOX_FDDB_FOLDS) if 'BoundingBox' in f]
    log.info("Extracting face images")
    extract_faces_from_images(bounding_box_files)
    log.info("Extracted face images")
    log.info("Extracting non-face images")
    extract_non_faces_from_images(bounding_box_files)
    log.info("Extracted non-face images")

    log.info("Reading extracted data")
    tr_face_data, tr_non_face_data, tr_face_labels, tr_non_face_labels, \
        te_face_data, te_non_face_data, te_face_labels, te_non_face_labels = data_preparation()
    log.info("Completed reading extracted data")

    log.info("Dumping extracted files")
    if not normalize:
        pickle.dump(tr_face_data, open(DIR_PATH + OUTPUT_DIR + "tr_face_data_not_norm.pkl", "wb"))
        pickle.dump(tr_non_face_data, open(DIR_PATH + OUTPUT_DIR + "tr_non_face_data_not_norm.pkl", "wb"))

        pickle.dump(te_face_data, open(DIR_PATH + OUTPUT_DIR + "te_face_data_not_norm.pkl", "wb"))
        pickle.dump(te_non_face_data, open(DIR_PATH + OUTPUT_DIR + "te_non_face_data_not_norm.pkl", "wb"))
    else:
        pickle.dump(tr_face_data, open(DIR_PATH + OUTPUT_DIR + "tr_face_data.pkl", "wb"))
        pickle.dump(tr_non_face_data, open(DIR_PATH + OUTPUT_DIR + "tr_non_face_data.pkl", "wb"))

        pickle.dump(te_face_data, open(DIR_PATH + OUTPUT_DIR + "te_face_data.pkl", "wb"))
        pickle.dump(te_non_face_data, open(DIR_PATH + OUTPUT_DIR + "te_non_face_data.pkl", "wb"))

    pickle.dump(tr_face_labels, open(DIR_PATH + OUTPUT_DIR + "tr_face_labels.pkl", "wb"))
    pickle.dump(tr_non_face_labels, open(DIR_PATH + OUTPUT_DIR + "tr_non_face_labels.pkl", "wb"))
    pickle.dump(te_face_labels, open(DIR_PATH + OUTPUT_DIR + "te_face_labels.pkl", "wb"))
    pickle.dump(te_non_face_labels, open(DIR_PATH + OUTPUT_DIR + "te_non_face_labels.pkl", "wb"))


if __name__ == '__main__':
    normalize = "True" == sys.argv[1] if len(sys.argv) == 2 else True
    main(normalize)
