import os, math, copy
# 動画処理用
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from tools import write_pose, write_rect
#libの中
CENTERNET_LIB_PATH = 'CenterNet/src/lib'
MODEL_PATH = 'CenterNet/models/multi_pose_dla_3x.pth'
import sys
sys.path.insert(0, CENTERNET_LIB_PATH)
from detectors.detector_factory import detector_factory
from opts import opts
# 受け取りと出力関係
from flask import Flask, request, redirect, url_for
# ファイル名をチェックする関数
from werkzeug.utils import secure_filename
# 画像のダウンロード
from flask import send_from_directory, render_template, flash

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = './uploads'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

TASK = 'multi_pose' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)

# ファイルを受け取る方法の指定
@app.route('/', methods=['GET', 'POST'])
def uploads_file():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        # データの取り出し
        file = request.files['file']
        # ファイル名がなかった時の処理
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        # ファイルのチェック
        if file:
            # 危険な文字を削除（サニタイズ処理）
            filename = secure_filename(file.filename)
            # ファイルの保存
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # アップロード後のページに転送
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template("index.html")

@app.route('/uploads/<filename>')
# ファイルを加工する
def uploaded_file(filename):
    VIDEO_PATH = app.config['UPLOAD_FOLDER']+'/'+filename
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #総フレーム数
    fps = round(cap.get(cv2.CAP_PROP_FPS)) #fps

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #画像高さ
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #画像幅
    size = (width, height) #画像サイズ

    #解析データの保存先
    out_path = app.config['UPLOAD_FOLDER']+'/'+'result.mp4'
    out = cv2.VideoWriter(
            out_path, 
            cv2.VideoWriter_fourcc('m','p','4', 'v'),
            fps,
            size
            )

    for _ in tqdm(range(10)):
        ret0, frame_read = cap.read()
        if not ret0:
            break
            
        ret = detector.run(frame_read)['results']
        
        for bbox in ret[1]:
            if bbox[4] > 0.5:
                points = np.array(bbox[5:39], dtype=np.int32).reshape(17, 2)
                write_pose(points,frame_read)
        
        out.write(frame_read)

    cap.release()
    out.release()
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run()