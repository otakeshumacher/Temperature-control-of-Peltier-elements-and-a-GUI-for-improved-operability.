import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# 📁 入力フォルダ（解析する動画があるフォルダ）
input_folder = r"D:\golf_kaiseki\trackingdata\20250227\straight"

# 📂 出力フォルダ（CSVと動画を分ける）
output_folder = r"D:\golf_kaiseki\trackingdata\20250227\straight\output"
csv_output_folder = os.path.join(output_folder, "csv")
video_output_folder = os.path.join(output_folder, "tracking_video_right")

# フォルダを作成（なければ作成）
os.makedirs(csv_output_folder, exist_ok=True)
os.makedirs(video_output_folder, exist_ok=True)

# MediaPipeのセットアップ
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 骨格ランドマークの名前
pose_landmarks_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder", "Left Elbow",
    "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle",
    "Right Ankle", "Left Heel", "Right Heel", "Left Foot Index", "Right Foot Index", "Left Toe", "Right Toe",
    "Left Eye Inner", "Left Eye Outer", "Right Eye Inner", "Right Eye Outer", "Left Cheek", "Right Cheek",
    "Left Jaw", "Right Jaw", "Left Temple", "Right Temple"
]

# 📁 指定フォルダ内のすべての動画を処理
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.mp4', '.mov', '.avi')):  # 対応する拡張子を指定
        video_path = os.path.join(input_folder, filename)

        # 📌 保存ファイル名（拡張子なし）
        base_name = os.path.splitext(filename)[0]

        # 📂 出力パス
        csv_path = os.path.join(csv_output_folder, f"{base_name}.csv")
        video_output_path = os.path.join(video_output_folder, f"{base_name}_tracking.mp4")

        # 動画を開く
        cap = cv2.VideoCapture(video_path)
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # 🎥 動画の保存設定
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

        # CSV用データリスト
        data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # 骨格を描画
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # 座標データを取得
                landmarks = results.pose_landmarks.landmark
                frame_data = [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks] + [landmark.z for landmark in landmarks]
                data.append(frame_data)

            # 動画を保存
            out.write(frame)

            # 表示（オプション）
            cv2.imshow('Swing Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 📝 CSVに保存
        df = pd.DataFrame(data)
        df.columns = [f"{name}_x" for name in pose_landmarks_names] + [f"{name}_y" for name in pose_landmarks_names] + [f"{name}_z" for name in pose_landmarks_names]
        df.to_csv(csv_path, index=False)

        print(f"✅ {filename} の解析完了！ → CSV: {csv_path}, 動画: {video_output_path}")
