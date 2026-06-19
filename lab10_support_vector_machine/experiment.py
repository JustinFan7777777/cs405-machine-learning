import cv2
import os
import pickle
from train import processFiles, trainSVM
from detector import Detector

# experiment.py 说明（中文注释）
# 本文件为实验/运行脚本，负责两件事：
# 1) 如果没有已保存的模型文件（MODEL_FILE），可以调用训练流程生成模型；
# 2) 加载模型并使用 `Detector` 在输入视频上运行检测，最终把检测结果写入输出视频文件。
#
# 文件的上方定义了一组超参数（滑动窗口参数、热图阈值、显示与导出选项），
# 用户可以通过修改这些常量来调节检测行为而无需深入代码实现。

# ============================================================================
# HYPERPARAMETERS - Adjust these to tune the detection pipeline
# ============================================================================

# Input/output video files
INPUT_VIDEO = "videos/test_video.mp4"
OUTPUT_VIDEO = "videos/result_video.mp4"
MODEL_FILE = "model.pkl"

# Sliding window parameters
WINDOW_SCALE = 1.5              # Scale factor for window size at each y level
X_OVERLAP = 0.5                 # Horizontal overlap between adjacent windows
Y_STEP = 0.05                   # Vertical step as fraction of image height
X_RANGE = (0.0, 1.0)            # Horizontal search area (0=left, 1=right)
Y_RANGE = (0.55, 0.95)          # Vertical search area (exclude sky at top)

# Heatmap/detection parameters
HEAT_THRESHOLD = 25             # Threshold for heatmap pixel values
HEATMAP_FRAMES = 10             # Number of frames to accumulate for smoothing
DRAW_HEATMAP = True             # Draw heatmap inset on output video
HEATMAP_INSET_SIZE = 0.15       # Size of heatmap as fraction of image
MIN_BBOX_SIZE = (50, 50)        # Minimum (width, height) for detection box

# Display parameters
SHOW_VIDEO = False              # Display video in real-time while processing
WRITE_FPS = 24                  # FPS for output video

# ============================================================================


def train_model_if_needed(model_file=MODEL_FILE):
    """
    Train the SVM classifier if model file doesn't exist.
    Uses the training hyperparameters from train.py.
    """
    if os.path.exists(model_file):
        print(f"Model file '{model_file}' already exists. Skipping training...")
        return

    print(f"Model file '{model_file}' not found. Training new model...")
    from train import (
        processFiles, trainSVM, POS_DIR, NEG_DIR, COLOR_SPACE,
        ORIENTATIONS, PIXELS_PER_CELL, CELLS_PER_BLOCK, HIST_BINS,
        SVM_C, HOG_LIB
    )

    print("Extracting features...")
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
        hist_bins=HIST_BINS
    )

    print("Training SVM classifier...")
    trainSVM(
        feature_data=feature_data,
        C=SVM_C,
        output_file=True,
        output_filename=model_file
    )
    print(f"Model saved to '{model_file}'")

    # 说明：此函数只是按需触发训练。如果数据集很大/训练耗时，
    # 建议在交互式会话或单独脚本中显式运行训练并保存模型，
    # 再在实验中直接加载已保存模型以节约时间。


def run_detection(model_file=MODEL_FILE, input_video=INPUT_VIDEO,
                  output_video=OUTPUT_VIDEO):
    """
    Load a trained SVM classifier and run vehicle detection on a video.

    Args:
        model_file (str): Path to saved classifier pickle file
        input_video (str): Path to input video file
        output_video (str): Path to save output video with detections
    """

    print("=" * 80)
    print("VEHICLE DETECTION USING HOG + SVM")
    print("=" * 80)

    # Check if input video exists
    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"Input video '{input_video}' not found.")

    # Check if model exists, train if needed
    if not os.path.isfile(model_file):
        print(f"Model file '{model_file}' not found!")
        train_model_if_needed(model_file)

    print(f"\nLoading model from '{model_file}'...")
    with open(model_file, "rb") as f:
        classifier_data = pickle.load(f)

    print("Initializing detector...")
    detector = Detector(
        init_size=(64, 64),
        x_overlap=X_OVERLAP,
        y_step=Y_STEP,
        x_range=X_RANGE,
        y_range=Y_RANGE,
        scale=WINDOW_SCALE
    )
    detector.loadClassifier(classifier_data=classifier_data)

    print(f"Opening video '{input_video}'...")
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file '{input_video}'")

    print("\nDetection Parameters:")
    print(f"  Heat Threshold: {HEAT_THRESHOLD}")
    print(f"  Heatmap Frames: {HEATMAP_FRAMES}")
    print(f"  Min Bbox Size: {MIN_BBOX_SIZE}")
    print(f"  Y-Range: {Y_RANGE} (exclude sky: {Y_RANGE[0]*100:.0f}% to {Y_RANGE[1]*100:.0f}%)")
    print(f"  Window Scale: {WINDOW_SCALE}")
    print(f"  X Overlap: {X_OVERLAP}")
    print(f"  Y Step: {Y_STEP}\n")

    print("Starting detection...")
    detector.detectVideo(
        video_capture=cap,
        num_frames=HEATMAP_FRAMES,
        threshold=HEAT_THRESHOLD,
        min_bbox=MIN_BBOX_SIZE,
        show_video=SHOW_VIDEO,
        draw_heatmap=DRAW_HEATMAP,
        draw_heatmap_size=HEATMAP_INSET_SIZE,
        write=True,
        write_fps=WRITE_FPS,
        output_filename=output_video
    )

    # 说明：detectVideo 会在结束后写出 output_video 文件（如果 write=True），
    # 并在内部使用滑动窗口、特征描述符、缩放器(scaler)与训练好的 SVM
    # 来对每个候选窗口进行判别，最终将检测框绘制在帧上。

    print("\n" + "=" * 80)
    print(f"Detection complete! Video saved to '{output_video}'")
    print("=" * 80)


if __name__ == "__main__":
    # Run the full detection pipeline
    run_detection(
        model_file=MODEL_FILE,
        input_video=INPUT_VIDEO,
        output_video=OUTPUT_VIDEO
    )



