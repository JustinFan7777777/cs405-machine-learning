#!/usr/bin/env python3
"""
Minimal training script to recreate `train.py` used in the project.

Functionality:
- Extract features using `Descriptor` from `descriptor.py` (HOG, color hist, spatial)
- Scale features with `StandardScaler`
- Train a `LinearSVC` and save `model.pkl` with classifier + scaler + params

This version is compact and self-contained for reproducibility.
"""
import os
import glob
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from descriptor import Descriptor


def convert_color(img, color_space):
    cs = (color_space or "BGR").lower()
    if cs == "rgb":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if cs in ("ycrcb", "ycrbc"):
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if cs == "hsv":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if cs == "luv":
        return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    if cs == "hls":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    return img


def find_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, "**", e), recursive=True))
    return sorted(files)


def extract_features(file_list, descriptor, color_space=None):
    feats = []
    for fp in tqdm(file_list, desc="features", unit="file"):
        img = cv2.imread(fp)
        if img is None:
            continue
        if color_space:
            img = convert_color(img, color_space)
        fv = descriptor.getFeatureVector(img)
        feats.append(fv)
    if len(feats) == 0:
        return np.zeros((0,))
    return np.vstack(feats)


def build_dataset(positive_dir, negative_dir, descriptor, color_space=None, test_size=0.2, val_size=0.1, random_state=42):
    pos_files = find_images(positive_dir)
    neg_files = find_images(negative_dir)

    X_pos = extract_features(pos_files, descriptor, color_space=color_space)
    X_neg = extract_features(neg_files, descriptor, color_space=color_space)

    y_pos = np.ones(X_pos.shape[0], dtype=int) if X_pos.size else np.array([], dtype=int)
    y_neg = np.zeros(X_neg.shape[0], dtype=int) if X_neg.size else np.array([], dtype=int)

    if X_pos.size == 0 and X_neg.size == 0:
        raise RuntimeError("No training data found in provided sample directories.")

    X = np.vstack((X_pos, X_neg))
    y = np.hstack((y_pos, y_neg))

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    X_train_val, X_test, y_train_val, y_test = train_test_split(Xs, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None)
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_rel, random_state=random_state, stratify=y_train_val if len(np.unique(y_train_val))>1 else None)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
    }


def train_and_save(data, C=10.0, model_path="model.pkl"):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    clf = LinearSVC(C=C, max_iter=20000, dual=False)
    t0 = time.time()
    clf.fit(X_train, y_train)
    t1 = time.time()

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"Trained LinearSVC (C={C}) in {t1 - t0:.1f}s — val acc: {acc:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))

    classifier_data = {
        "classifier": clf,
        "scaler": data["scaler"],
    }
    with open(model_path, "wb") as fh:
        pickle.dump(classifier_data, fh)

    print(f"Saved model to {model_path}")
    return clf


def main():
    # Example defaults — adjust paths if your dataset is elsewhere
    pos_dir = os.path.join("samples", "vehicles")
    neg_dir = os.path.join("samples", "non-vehicles")

    params = {
        "size": (64, 64),
        "hog_bins": 9,
        "pix_per_cell": (8, 8),
        "cells_per_block": (2, 2),
        "hist_bins": 16,
        "spatial_size": (16, 16),
        "hog_features": True,
        "hist_features": True,
        "spatial_features": True,
    }

    descriptor = Descriptor(hog_features=params["hog_features"],
                            hist_features=params["hist_features"],
                            spatial_features=params["spatial_features"],
                            size=params["size"],
                            hog_bins=params["hog_bins"],
                            pix_per_cell=params["pix_per_cell"],
                            cells_per_block=params["cells_per_block"],
                            hist_bins=params["hist_bins"],
                            spatial_size=params["spatial_size"])

    print("Building dataset (this may take a while)...")
    data = build_dataset(pos_dir, neg_dir, descriptor, color_space="ycrcb")

    print("Training classifier...")
    train_and_save(data, C=10.0, model_path="model.pkl")


if __name__ == "__main__":
    main()
from datetime import datetime
import os
import pickle
import random
import time
import warnings
import cv2
import tqdm
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from descriptor import Descriptor

# ============================================================================
# HYPERPARAMETERS - Adjust these to tune the model
# ============================================================================
COLOR_SPACE = "ycrcb"      # Color space: "bgr", "gray", "hls", "hsv", "lab", "luv", "ycrcb", "yuv"
ORIENTATIONS = 9           # HOG orientations (6, 9, or 12)
PIXELS_PER_CELL = (8, 8)   # HOG pixels per cell
CELLS_PER_BLOCK = (2, 2)   # HOG cells per block
HIST_BINS = 16             # Color histogram bins
SPATIAL_SIZE = (16, 16)    # Spatial binning size
SVM_C = 10.0               # SVM regularization parameter (larger = stricter)
HOG_LIB = "sk"             # HOG library: "cv" (OpenCV) or "sk" (scikit-image)

# File paths
POS_DIR = "samples/vehicles"
NEG_DIR = "samples/non-vehicles"
OUTPUT_CLASSIFIER = "model.pkl"  # Saved classifier file
# ============================================================================


def processFiles(pos_dir, neg_dir, recurse=False, output_file=False,
                 output_filename=None, color_space="bgr", channels=[0, 1, 2],
                 hog_features=False, hist_features=False, spatial_features=False,
                 hog_lib="cv", size=(64, 64), hog_bins=9, pix_per_cell=(8, 8),
                 cells_per_block=(2, 2), block_stride=None, block_norm="L1",
                 transform_sqrt=True, signed_gradient=False, hist_bins=16,
                 spatial_size=(16, 16)):
    """
    Extract features from positive samples and negative samples.
    Store feature vectors in a dict and optionally save to pickle file.

    @param pos_dir (str): Path to directory containing positive samples.
    @param neg_dir (str): Path to directory containing negative samples.
    @param recurse (bool): Traverse directories recursively (else, top-level only).
    @param output_file (bool): Save processed samples to file.
    @param output_filename (str): Output file filename.
    @param color_space (str): Color space conversion.
    @param channels (list): Image channel indices to use.
    
    For remaining arguments, refer to Descriptor class:
    @see descriptor.Descriptor#__init__(...)

    @return feature_data (dict): Lists of sample features split into training,
        validation, test sets; scaler object; parameters used to
        construct descriptor and process images.

    NOTE: OpenCV HOGDescriptor currently only supports 1-channel and 3-channel
    images, not 2-channel images.
    """

    if not (hog_features or hist_features or spatial_features):
        raise RuntimeError("No features selected (set hog_features=True, "
                           + "hist_features=True, and/or spatial_features=True.)")

    pos_dir = os.path.abspath(pos_dir)
    neg_dir = os.path.abspath(neg_dir)

    if not os.path.isdir(pos_dir):
        raise FileNotFoundError("Directory " + pos_dir + " does not exist.")
    if not os.path.isdir(neg_dir):
        raise FileNotFoundError("Directory " + neg_dir + " does not exist.")

    print("Building file list...")
    if recurse:
        pos_files = [os.path.join(rootdir, file) for rootdir, _, files
                     in os.walk(pos_dir) for file in files]
        neg_files = [os.path.join(rootdir, file) for rootdir, _, files
                     in os.walk(neg_dir) for file in files]
    else:
        pos_files = [os.path.join(pos_dir, file) for file in
                     os.listdir(pos_dir) if os.path.isfile(os.path.join(pos_dir, file))]
        neg_files = [os.path.join(neg_dir, file) for file in
                     os.listdir(neg_dir) if os.path.isfile(os.path.join(neg_dir, file))]

    print("{} positive files and {} negative files found.\n".format(
        len(pos_files), len(neg_files)))

    # Get color space information.
    color_space = color_space.lower()
    if color_space == "gray":
        color_space_name = "grayscale"
        cv_color_const = cv2.COLOR_BGR2GRAY
        channels = [0]
    elif color_space == "hls":
        color_space_name = "HLS"
        cv_color_const = cv2.COLOR_BGR2HLS
    elif color_space == "hsv":
        color_space_name = "HSV"
        cv_color_const = cv2.COLOR_BGR2HSV
    elif color_space == "lab":
        color_space_name = "Lab"
        cv_color_const = cv2.COLOR_BGR2Lab
    elif color_space == "luv":
        color_space_name = "Luv"
        cv_color_const = cv2.COLOR_BGR2Luv
    elif color_space == "ycrcb" or color_space == "ycc":
        color_space_name = "YCrCb"
        cv_color_const = cv2.COLOR_BGR2YCrCb
    elif color_space == "yuv":
        color_space_name = "YUV"
        cv_color_const = cv2.COLOR_BGR2YUV
    else:
        color_space_name = "BGR"
        cv_color_const = -1

    # Get names of desired features.
    features = [feature_name for feature_name, feature_bool
                in zip(["HOG", "color histogram", "spatial"],
                       [hog_features, hist_features, spatial_features])
                if feature_bool == True]

    feature_str = features[0]
    for feature_name in features[1:]:
        feature_str += ", " + feature_name

    # Get information about channel indices.
    if len(channels) == 2 and hog_features and hog_lib == "cv":
        warnings.warn("OpenCV HOG does not support 2-channel images",
                      RuntimeWarning)

    channel_index_str = str(channels[0])
    for ch_index in channels[1:]:
        channel_index_str += ", {}".format(ch_index)

    print("Converting images to " + color_space_name + " color space and "
          + "extracting " + feature_str + " features from channel(s) "
          + channel_index_str + ".\n")

    # Store feature vectors for positive samples in list pos_features and
    # for negative samples in neg_features.
    pos_features = []
    neg_features = []
    start_time = time.time()

    # Get feature descriptor object to call on each sample.
    descriptor = Descriptor(hog_features=hog_features, hist_features=hist_features,
                            spatial_features=spatial_features, hog_lib=hog_lib, size=size,
                            hog_bins=hog_bins, pix_per_cell=pix_per_cell,
                            cells_per_block=cells_per_block, block_stride=block_stride,
                            block_norm=block_norm, transform_sqrt=transform_sqrt,
                            signed_gradient=signed_gradient, hist_bins=hist_bins,
                            spatial_size=spatial_size)

    # Iterate through files and extract features.
    bar = tqdm.tqdm(total=len(pos_files) + len(neg_files), desc="Convert process")
    for i, filepath in enumerate(pos_files + neg_files):
        image = cv2.imread(filepath)
        try:
            if image is None or image.size == 0:
                bar.update()
                continue

            if cv_color_const > -1:
                image = cv2.cvtColor(image, cv_color_const)

            if len(image.shape) > 2:
                image = image[:, :, channels]

            feature_vector = descriptor.getFeatureVector(image)

            if i < len(pos_files):
                pos_features.append(feature_vector)
            else:
                neg_features.append(feature_vector)
        except Exception as e:
            pass
        bar.update()
    bar.close()

    print("Features extracted from {} files in {:.1f} seconds\n".format(
        len(pos_features) + len(neg_features), time.time() - start_time))

    # Store the length of the feature vector produced by the descriptor.
    num_features = len(pos_features[0])
    ##########################################Answer Area 1 begin#############################################
    # TODO: Instantiate scaler and scale pos and neg features.
    # 说明：这里使用 sklearn 的 StandardScaler 对所有样本特征做均值-方差归一化。
    # 归一化是训练线性分类器（如 LinearSVC）时常用的预处理步骤，可以提升
    # 收敛速度并避免某些特征尺度过大支配分类结果。
    print("Instantiate scaler and scale features.\n")
    scaler = StandardScaler().fit(pos_features + neg_features)
    pos_features = scaler.transform(pos_features)
    neg_features = scaler.transform(neg_features)
    ##########################################Answer Area 1 end#############################################

    # validation, and test sets.
    print("Shuffling samples into training, cross-validation, and test sets.\n")
    random.shuffle(pos_features)
    random.shuffle(neg_features)
    
    # Use pos_train, pos_val, pos_test and neg_train, neg_val, neg_test to represent 
    # the Train, Validation and Test sets of Positive and Negative sets.
    ##########################################Answer Area 2 begin#############################################
    # TODO: Split 75/20/5 into training.
    # pos_train = neg_train = pos_val = neg_val = pos_test = neg_test = None
    # pass
    # 这里将正负样本分别按 75% / 20% / 5% 的比例划分为训练/验证/测试集。
    # 具体实现：先计算训练集样本个数，再按索引切片得到三组。
    # 注意：这种简单切分仍保持类别内互相独立，但不保证跨正负样本的严格
    # 随机混合（我们已经对各自列表做了 shuffle）。对于更严格的划分，可
    # 使用 sklearn.model_selection.train_test_split 并在两个类别上做分层采样。
    num_pos_train = int(round(0.75 * len(pos_features)))
    num_neg_train = int(round(0.75 * len(neg_features)))

    num_pos_val = int(round(0.2 * len(pos_features)))
    num_neg_val = int(round(0.2 * len(neg_features)))

    pos_train = pos_features[0 : num_pos_train]
    neg_train = neg_features[0 : num_neg_train]

    pos_val = pos_features[num_pos_train : (num_pos_train + num_pos_val)]
    neg_val = neg_features[num_neg_train : (num_neg_train + num_neg_val)]

    pos_test = pos_features[(num_pos_train + num_pos_val):]
    neg_test = neg_features[(num_neg_train + num_neg_val):]
    ##########################################Answer Area 2 end#############################################
    # Store sample data and parameters in dict.
    # Descriptor class object seems to produce errors when unpickling and
    # has been commented out below. The descriptor will be re-instantiated
    # by the Detector object later.
    feature_data = {
        "pos_train": pos_train,
        "neg_train": neg_train,
        "pos_val": pos_val,
        "neg_val": neg_val,
        "pos_test": pos_test,
        "neg_test": neg_test,
        # "descriptor": descriptor,
        "scaler": scaler,
        "hog_features": hog_features,
        "hist_features": hist_features,
        "spatial_features": spatial_features,
        "color_space": color_space,
        "cv_color_const": cv_color_const,
        "channels": channels,
        "hog_lib": hog_lib,
        "size": size,
        "hog_bins": hog_bins,
        "pix_per_cell": pix_per_cell,
        "cells_per_block": cells_per_block,
        "block_stride": block_stride,
        "block_norm": block_norm,
        "transform_sqrt": transform_sqrt,
        "signed_gradient": signed_gradient,
        "hist_bins": hist_bins,
        "spatial_size": spatial_size,
        "num_features": num_features
    }

    # Pickle to file if desired.
    if output_file:
        if output_filename is None:
            output_filename = (datetime.now().strftime("%Y%m%d%H%M")
                               + "_data.pkl")

        pickle.dump(feature_data, open(output_filename, "wb"))
        print("Sample and parameter data saved to {}\n".format(output_filename))

    return feature_data


def trainSVM(filepath=None, feature_data=None, C=1,
             loss="squared_hinge", penalty="l2", dual=False, fit_intercept=False,
             output_file=False, output_filename=None):
    """
        Train a classifier from feature data extracted by processFiles().

        @param filepath (str): Path to feature data pickle file.
        @param feature_data (dict): Feature data dict returned by processFiles().
            NOTE: Either a file or dict may be supplied.
        @param output_file (bool): Save classifier and parameters to file.
        @param output_filename (str): Name of output file.

        For remaining arguments, @see sklearn.svm.LinearSVC()

        @return classifier_data (dict): Dict containing trained classifier and
            relevant training/processing feature parameters.
    """

    print("Loading sample data.")
    if filepath is not None:
        filepath = os.path.abspath(filepath)
        if not os.path.isfile(filepath):
            raise FileNotFoundError("File " + filepath + " does not exist.")
        feature_data = pickle.load(open(filepath, "rb"))
    elif feature_data is None:
        raise ValueError("Invalid feature data supplied.")
    ##########################################Answer Area 3 begin#############################################
    # TODO: Train classifier on training set, using sklearn LinearSVC model.
    #      Use validation sets to adjust your algorithm.
    #      Run your classifier on the test sets and output the accuracy,
    #      precision, recall and F-1 score.

    # 将 feature_data 中的各个集合转换为 numpy 数组，方便后续堆叠与训练
    pos_train = np.asarray(feature_data["pos_train"])
    neg_train = np.asarray(feature_data["neg_train"])
    pos_val = np.asarray(feature_data["pos_val"])
    neg_val = np.asarray(feature_data["neg_val"])
    pos_test = np.asarray(feature_data["pos_test"])
    neg_test = np.asarray(feature_data["neg_test"])

    # 把正负训练样本竖直堆叠为单个训练集，并构建对应标签：正样本为 1，负样本为 0
    train_set = np.vstack((pos_train, neg_train))
    train_labels = np.concatenate((np.ones(pos_train.shape[0],), np.zeros(neg_train.shape[0],)))
    print("Training Phase.\n")
    start_time = time.time()

    # 使用 sklearn 的 LinearSVC 训练线性支持向量机。参数 C 控制正则化强度，
    # dual=False 在样本数量多于特征数量时更快；loss/penalty 等参数可用于调优。
    classifier = svm.LinearSVC(C=C, loss=loss, penalty=penalty, dual=dual, fit_intercept=fit_intercept)
    classifier.fit(train_set, train_labels)

    # 在验证集上评估性能（这里分别对正/负验证集做预测并计算精度/召回/F1）
    print("Validation Phase.\n")
    pos_val_predicted = classifier.predict(pos_val)
    neg_val_predicted = classifier.predict(neg_val)
    # 计算混淆矩阵元素并据此推导指标
    false_neg_val = np.sum(pos_val_predicted != 1)
    false_pos_val = np.sum(neg_val_predicted == 1)
    fp = np.sum(pos_val_predicted != 1)
    fn = np.sum(neg_val_predicted == 1)
    tp = pos_val.shape[0] - fp
    tn = neg_val.shape[0] - fn
    accuracy = (tp + tn) / (fp + fn + tp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 / ((1 / precision) + (1 / recall))
    print("Validation Accuracy: ", accuracy)
    print("Validation Precision: ", precision)
    print("Validation Recall: ", recall)
    print("Validation F-1 Score: ", f1_score)

    # 简单的微调策略：把被分类错误（假负）的验证样本加入训练集中，
    # 并把误报（假正）的验证样本加入负样本训练集中，随后重新训练。
    # 这种做法有点类似于 hard-negative mining，可以提升模型对困难样本的鲁棒性。
    pos_train = np.vstack((pos_train, pos_val[pos_val_predicted != 1, :]))
    neg_train = np.vstack((neg_train, neg_val[neg_val_predicted == 1, :]))
    train_set = np.vstack((pos_train, neg_train))
    train_labels = np.concatenate((np.ones(pos_train.shape[0],), np.zeros(neg_train.shape[0],)))
    classifier.fit(train_set, train_labels)

    # 在测试集上评估最终模型性能并输出指标
    print("Testing Phase.\n")
    pos_test_predicted = classifier.predict(pos_test)
    neg_test_predicted = classifier.predict(neg_test)
    false_neg_test = np.sum(pos_test_predicted != 1)
    false_pos_test = np.sum(neg_test_predicted == 1)
    fp = np.sum(pos_test_predicted != 1)
    fn = np.sum(neg_test_predicted == 1)
    tp = pos_test.shape[0] - fp
    tn = neg_test.shape[0] - fn
    accuracy = (tp + tn) / (fp + fn + tp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 / ((1 / precision) + (1 / recall))
    print("Testing Accuracy: ", accuracy)
    print("Testing Precision: ", precision)
    print("Testing Recall: ", recall)
    print("Testing F-1 Score: ", f1_score)
    ##########################################Answer Area 3 end#############################################
    # Store classifier data and parameters in new dict that excludes
    # sample data from feature_data dict.
    excludeKeys = ("pos_train", "neg_train", "pos_val", "neg_val",
                   "pos_test", "neg_test")
    # 将不必要的大样本数据（训练/验证/测试向量）从要保存的字典中剔除，
    # 只保留缩放器、描述符参数和模型本身等用于推理所需的信息。
    classifier_data = {key: val for key, val in feature_data.items()
                       if key not in excludeKeys}
    ##########################################Answer Area 4 begin#############################################
    # classifier_data["classifier"] = None  # TODO: complement the assignment state with the name of your classifier
    # 把训练好的分类器对象放入返回/保存的字典中，便于后续通过
    # Detector.loadClassifier() 恢复推理环境。
    classifier_data["classifier"] = classifier
    ##########################################Answer Area 4 end#############################################
    if output_file:
        if output_filename is None:
            output_filename = (datetime.now().strftime("%Y%m%d%H%M")
                               + "_classifier.pkl")

        pickle.dump(classifier_data, open(output_filename, "wb"))
        print("\nSVM classifier data saved to {}".format(output_filename))

    return classifier_data


if __name__ == "__main__":
    """
    Main entry point for training the SVM classifier.
    Uses hyperparameters defined at the top of the file.
    Saves the trained classifier to OUTPUT_CLASSIFIER.
    """
    print("=" * 80)
    print("TRAINING SVM CLASSIFIER FOR VEHICLE DETECTION")
    print("=" * 80)
    print(f"\nHyperparameters:")
    print(f"  Color Space: {COLOR_SPACE}")
    print(f"  Orientations: {ORIENTATIONS}")
    print(f"  Pixels per Cell: {PIXELS_PER_CELL}")
    print(f"  Cells per Block: {CELLS_PER_BLOCK}")
    print(f"  HOG Library: {HOG_LIB}")
    print(f"  SVM C: {SVM_C}\n")

    # Extract features from positive and negative samples
    feature_data = processFiles(
        pos_dir=POS_DIR,
        neg_dir=NEG_DIR,
        recurse=True,
        color_space=COLOR_SPACE,
        hog_features=True,
        hist_features=True,
        spatial_features=True,
        hog_lib=HOG_LIB,
        hog_bins=ORIENTATIONS,
        pix_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        hist_bins=HIST_BINS,
        spatial_size=SPATIAL_SIZE
    )

    # Train SVM classifier
    print("\n" + "=" * 80)
    classifier_data = trainSVM(
        feature_data=feature_data,
        C=SVM_C,
        output_file=True,
        output_filename=OUTPUT_CLASSIFIER
    )

    print("=" * 80)
    print("Training complete! Model saved to:", OUTPUT_CLASSIFIER)
    print("=" * 80)
