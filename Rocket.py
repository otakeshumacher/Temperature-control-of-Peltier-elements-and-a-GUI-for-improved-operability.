# 必要なライブラリのインポート
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sktime.transformations.panel.rocket import Rocket
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog

# Optunaのログ出力をINFOレベルに設定
optuna.logging.set_verbosity(optuna.logging.INFO)

print("データの読み込みと前処理を開始します...")

# GUIでデータフォルダのパスを選択する関数
def select_folder(title):
    try:
        root = tk.Tk()
        root.withdraw()  # メインウィンドウを表示しない
        folder_path = filedialog.askdirectory(title=title)
        root.destroy()  # ダイアログを閉じた後、tkinterのインスタンスを破棄
        return folder_path
    except tk.TclError:
        print("GUIが利用できない環境のため、フォルダパスを直接入力してください。")
        return input(f"{title} (パス): ")

# データ拡張の有無をユーザーに確認する
use_data_augmentation = input("データ拡張を行いますか？ (yes/no): ").lower() == 'yes'
if use_data_augmentation:
    print("データ拡張を行います。")
else:
    print("データ拡張を行いません。")

# データフォルダのパスをユーザーに選択させる
print("データフォルダを選択してください (動物フォルダを含む親フォルダ)。")
data_folder = select_folder("データフォルダを選択")

# ファイル選択のための文字列入力をユーザーに促す関数
def get_file_string_input(prompt):
    return input(prompt)

# トレーニングデータのファイル選択文字列をユーザーに指定させる
print("トレーニングデータの設定:")
training_label_0_file_string = get_file_string_input("トレーニングデータ（ラベル0）に指定するファイルに含まれる文字列 (例: pre0_): ")
training_label_1_file_string = get_file_string_input("トレーニングデータ（ラベル1）に指定するファイルに含まれる文字列 (例: D4 または D3): ")

# テストデータのファイル選択文字列をユーザーに指定させる
print("テストデータの設定:")
test_label_0_file_string = get_file_string_input("テストデータ（ラベル0）に指定するファイルに含まれる文字列 (例: pre_): ")
test_label_1_file_string = get_file_string_input("テストデータ（ラベル1）に指定するファイルに含まれる文字列 (例: D7): ")

# CSVファイルを読み込む関数

def load_data_from_folder(base_folder, file_string, expected_shape=None):
    """
    複数のフォルダ内にあるCSVファイルを読み込む関数。

    Args:
        base_folder (str): データが格納されている親ディレクトリ
        file_string (str): ファイル名に含まれる識別文字列
        expected_shape (tuple, optional): 期待するデータ形状（最初のファイルで決定）

    Returns:
        data_list (list): 読み込んだデータ（NumPy配列のリスト）
        expected_shape (tuple): 使用されたデータ形状
    """
    data_list = []
    
    # base_folder 内のすべてのサブフォルダを探索
    all_folders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    
    # 親ディレクトリ自体も対象にする
    all_folders.append(base_folder)

    for folder_path in all_folders:
        file_paths = glob.glob(os.path.join(folder_path, "*.csv"))
        selected_files = [file for file in file_paths if file_string in file]

        if not selected_files:
            print(f"警告: {folder_path} 内に '{file_string}' を含むファイルが見つかりませんでした。")
            continue

        print(f"フォルダ {folder_path} から '{file_string}' を含む {len(selected_files)} 個のファイルを読み込んでいます...")

        for file in tqdm(selected_files, desc=f"{folder_path} から '{file_string}' を含むファイルの読み込み中"):
            try:
                data = pd.read_csv(file, skiprows=1, usecols=range(0, 99))
                if expected_shape is None:
                    expected_shape = data.shape
                    print(f"最初のファイルからデータ形状を決定： {expected_shape}")
                    data_list.append(data.values)
                elif data.shape == expected_shape:
                    data_list.append(data.values)
                else:
                    print(f"ファイル {file} の形状が期待と異なります: {data.shape}, スキップします...")

            except Exception as e:
                print(f"ファイル {file} の読み込みに失敗しました: {e}, スキップします...")

    if not data_list:
        print(f"エラー: {file_string} を含むデータが読み込めませんでした。")
        return data_list, expected_shape

    return data_list, expected_shape


def get_optimal_cv_splits(n_samples, n_classes):
    """
    データ数とクラス数に基づいて最適なクロスバリデーションの分割数を決定する
    """
    if n_samples < 20:
        return 2  # データが非常に少ない場合
    elif n_samples < 50:
        return 3  # データが少ない場合
    elif n_samples < 100:
        return 5  # データが中程度の場合
    else:
        return min(10, n_samples // (n_classes * 10))  # データが多い場合、各分割で少なくとも10サンプル×クラス数を確保

# データ拡張（ノイズ付きデータの追加）
def augment_data(X, y, noise_level=0.05):
    augmented_X = []
    augmented_y = []
    for i in tqdm(range(len(X)), desc="データを拡張中"):
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        noisy_series = X[i] + np.random.normal(0, noise_level, X[i].shape)
        augmented_X.append(noisy_series)
        augmented_y.append(y[i])
    return np.array(augmented_X), np.array(augmented_y, dtype=int)

# 結果を保存するファイルを開く
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_filename = f"{script_name}_results.txt"
with open(output_filename, "w") as f:
    
    # 各動物フォルダを処理
    f1_scores = []
    animal_folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    
    for animal_folder_name in animal_folders:
        animal_folder_path = os.path.join(data_folder, animal_folder_name)
        print(f"\n処理中の動物フォルダ: {animal_folder_path}")

        # トレーニングデータの読み込み
        print("ラベル0のトレーニングデータを読み込んでいます...")
        label_0_data, expected_shape = load_data_from_folder(animal_folder_path, training_label_0_file_string)
        if not label_0_data:  # label_0_data が空の場合は、以降の処理をスキップ
            print(f"エラー: ラベル0のトレーニングデータが読み込めなかったため、{animal_folder_name} をスキップします。")
            continue

        print("ラベル1のトレーニングデータを読み込んでいます...")
        
        # D3とD4のどちらをラベル1とするかを選択
        if training_label_1_file_string == "D4":
            label_1_data, _ = load_data_from_folder(animal_folder_path, "D4", expected_shape)
            if not label_1_data:
                label_1_data, _ = load_data_from_folder(animal_folder_path, "D3", expected_shape)  # D4がない場合はD3を試す
                print("D4ファイルが見つからなかったため、D3ファイルをラベル1として使用します。")
        elif training_label_1_file_string == "D3":
             label_1_data, _ = load_data_from_folder(animal_folder_path, "D3", expected_shape)
             if not label_1_data:
                label_1_data, _ = load_data_from_folder(animal_folder_path, "D4", expected_shape)  # D3がない場合はD4を試す
                print("D3ファイルが見つからなかったため、D4ファイルをラベル1として使用します。")
        else:
             label_1_data, _ = load_data_from_folder(animal_folder_path, training_label_1_file_string, expected_shape)
        
        if not label_1_data:
                print("エラー：ラベル1のトレーニングデータがありません。スキップします。")
                continue

        # データの整形
        print("トレーニングデータを結合しています...")
        X_train = np.concatenate([np.array(label_0_data), np.array(label_1_data)])
        y_train = np.concatenate([np.zeros(len(label_0_data)), np.ones(len(label_1_data))])
        
        if expected_shape is None:
            print("エラー：データ形状が検出できませんでした。プログラムを終了します。")
            continue

        n_timepoints = expected_shape[0]
        n_variables = expected_shape[1]

        # データの形状を修正
        print("トレーニングデータの形状を修正しています...")
        X_train_reshaped = X_train.reshape((X_train.shape[0], n_timepoints, n_variables))

        # トレーニングデータのデータ拡張
        if use_data_augmentation:
            print("トレーニングデータにデータ拡張を適用しています...")
            X_train_aug, y_train_aug = augment_data(X_train_reshaped, y_train)
            y_train_aug = y_train_aug.astype(int)
        else:
            print("トレーニングデータにデータ拡張を行いません...")
            X_train_aug = X_train_reshaped
            y_train_aug = y_train.astype(int)
        
        # ROCKET変換
        print("トレーニングデータにROCKET変換を適用しています...")
        rocket = Rocket(random_state=42)
        X_train_augmented_transformed = rocket.fit_transform(X_train_aug)

        # 特徴量の数を確認
        n_features = X_train_augmented_transformed.shape[1]
        print(f"ROCKET変換後の特徴量の数: {n_features}")

        # 特徴選択のためのkを設定
        k = min(50000, n_features)  # 希望する特徴量の数（最大50000）

        # 特徴選択
        print(f"特徴選択を行っています (上位{k}個)...")
        selector = SelectKBest(f_classif, k=k)
        X_train_augmented_selected = selector.fit_transform(X_train_augmented_transformed, y_train_aug)

        # データのスケーリング
        print("トレーニングデータをスケーリングしています...")
        scaler = StandardScaler()
        X_train_augmented_scaled = scaler.fit_transform(X_train_augmented_selected)

        n_samples = len(X_train_augmented_scaled)
        n_classes = len(set(y_train_aug))
        optimal_splits = get_optimal_cv_splits(n_samples, n_classes)

        print(f"データ拡張後のトレーニングデータ数: {len(X_train_augmented_scaled)}")
        print(f"拡張後のトレーニングデータ形状: {X_train_augmented_scaled.shape}")
        print(f"各クラスのサンプル数: {np.bincount(y_train_aug)}")
        print(f"最適なクロスバリデーション分割数: {optimal_splits}")
        
        # Optunaを使ってRandomForestの最適なハイパーパラメータを最適化する
        print("Optunaによるハイパーパラメータの最適化を開始します...")
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 100, 1000) #探索範囲を広げた
            max_depth = trial.suggest_int("max_depth", 5, 50, log=True) #探索範囲を広げた
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            
            # クロスバリデーションを使用して精度を評価
            cv = StratifiedKFold(n_splits=optimal_splits, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X_train_augmented_scaled, y_train_aug, cv=cv, scoring="f1_macro")
            return scores.mean()

        # コールバック関数で試行の結果を表示
        def callback(study, trial):
            print(f"試行 {trial.number}: F1スコア {trial.value} using {trial.params}")

        # Optunaのスタディを作成し、最適化を実行
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50, callbacks=[callback])  # Optunaの試行回数を設定50

        # 最適な分類器を選択
        print("ハイパーパラメータの最適化が完了しました。")
        best_trial = study.best_trial
        print(f"最適な試行番号: {best_trial.number}")
        print(f"最適なハイパーパラメータ: {best_trial.params}")

        # 最適なモデルを再訓練
        print("最適なハイパーパラメータでモデルを再訓練しています...")
        best_clf = RandomForestClassifier( # 直接RandomForestClassifierを定義
            n_estimators=best_trial.params["n_estimators"],
            max_depth=best_trial.params["max_depth"], random_state=42)

        # 全訓練データでモデルを再訓練
        best_clf.fit(X_train_augmented_scaled, y_train_aug)

        # テストデータの読み込み
        print("テストデータを読み込んでいます...")
        print("ラベル0のテストデータを読み込んでいます...")
        label_0_test_data, _ = load_data_from_folder(animal_folder_path, test_label_0_file_string, expected_shape)
        if not label_0_test_data:
            print(f"エラー：ラベル0のテストデータが読み込めませんでした。{animal_folder_name}をスキップします")
            continue
        print("ラベル1のテストデータを読み込んでいます...")
        
        # D3とD4のどちらをラベル1とするかを選択
        if test_label_1_file_string == "D4":
            label_1_test_data, _ = load_data_from_folder(animal_folder_path, "D4", expected_shape)
            if not label_1_test_data:
                label_1_test_data, _ = load_data_from_folder(animal_folder_path, "D3", expected_shape) # D4がない場合はD3を試す
                print("D4ファイルが見つからなかったため、D3ファイルをラベル1として使用します。")
        elif test_label_1_file_string == "D3":
            label_1_test_data, _ = load_data_from_folder(animal_folder_path, "D3", expected_shape)
            if not label_1_test_data:
                label_1_test_data, _ = load_data_from_folder(animal_folder_path, "D4", expected_shape)  # D3がない場合はD4を試す
                print("D3ファイルが見つからなかったため、D4ファイルをラベル1として使用します。")
        else:
            label_1_test_data, _ = load_data_from_folder(animal_folder_path, test_label_1_file_string, expected_shape)
        
        if not label_1_test_data:
                print("エラー：ラベル1のテストデータがありません。スキップします。")
                continue
        
        # テストデータをNumPy配列に変換
        print("テストデータを結合しています...")
        X_test = np.concatenate([np.array(label_0_test_data), np.array(label_1_test_data)])
        y_test = np.concatenate([np.zeros(len(label_0_test_data)), np.ones(len(label_1_test_data))])

        # テストデータの形状を修正
        print("テストデータの形状を修正しています...")
        X_test_reshaped = X_test.reshape((X_test.shape[0], n_timepoints, n_variables))
        
        # テストデータのデータ拡張
        print("テストデータにデータ拡張を適用しません...")
        X_test_aug = X_test_reshaped
        y_test_aug = y_test.astype(int)

        # テストデータのROCKET変換
        print("テストデータにROCKET変換を適用しています...")
        X_test_augmented_transformed = rocket.transform(X_test_aug)

        # 特徴選択とスケーリングをテストデータに適用
        print("テストデータの特徴選択とスケーリングを行っています...")
        X_test_augmented_selected = selector.transform(X_test_augmented_transformed)
        X_test_augmented_scaled = scaler.transform(X_test_augmented_selected)

        # テストデータで予測を行っています...
        print("テストデータで予測を行っています...")
        y_pred_test = best_clf.predict(X_test_augmented_scaled)

        # F1スコアの計算と表示
        f1_test = f1_score(y_test_aug, y_pred_test, average='macro')
        print(f"テストデータのF1スコア: {f1_test * 100:.2f}%")
        f1_scores.append(f1_test)

        # 混同行列の計算
        print("混同行列を作成・保存します...")
        conf_matrix = confusion_matrix(y_test_aug, y_pred_test)
        print("混同行列:")
        print(conf_matrix)

        # 混同行列の可視化と保存
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot()
        plt.title("Confusion Matrix")
        # 混同行列をファイルとして保存
        confusion_matrix_filename = f"{script_name}_{animal_folder_name}.png"
        plt.savefig(confusion_matrix_filename)
        plt.close()

        # 実際のラベルと予測ラベルの比較を可視化して保存する関数
        def plot_and_save_actual_vs_predicted(y_true, y_pred, f1, filename):
            plt.figure(figsize=(10, 6))
            plt.plot(y_true, label="Actual Labels", color="blue", alpha=0.6)
            plt.plot(y_pred, label="Predicted Labels", color="orange", alpha=0.6)
            plt.title(f"Comparison of Actual and Predicted Labels\nF1 Score: {f1:.4f}")
            plt.xlabel("Test Instances")
            plt.ylabel("Label")
            plt.legend()
            plt.savefig(filename)
            plt.close()  # メモリ解放のために図を閉じる

        # 実際のラベルと予測ラベルの比較をプロットして保存
        comparison_filename = f"{script_name}_{animal_folder_name}_2.png"
        plot_and_save_actual_vs_predicted(y_test_aug, y_pred_test, f1_test, comparison_filename)

        print(f"混同行列を {confusion_matrix_filename} として保存しました")
        print(f"ラベル比較グラフを {comparison_filename} として保存しました")
        f.write(f"動物フォルダ: {animal_folder_name}\n")
        f.write(f"   F1スコア: {f1_test * 100:.2f}%\n")
    
    # 全ての動物の平均F1スコアを計算して出力
    average_f1 = np.mean(f1_scores) if f1_scores else 0
    f.write(f"\n全ての動物の平均F1スコア: {average_f1 * 100:.2f}%\n")

print(f"結果を {output_filename} に保存しました。")