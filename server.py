import gc
import os
import shutil
import time
import uuid
from threading import Lock

import torch
from flask import Flask, session, request, send_from_directory
from flask_apscheduler import APScheduler

from utools.mask import generate_masks, draw_with_masks
from utools.validator import allowed_file

app = Flask(__name__, static_folder='static')
UPLOAD = 'upload'
CHECKPOINT = './checkpoints/sam_vit_h_4b8939.pth'
app.config['UPLOAD'] = UPLOAD
app.config['CHECKPOINT'] = CHECKPOINT

# 生成24位的随机数种子，用于产生SESSION ID
app.secret_key = os.urandom(24)

# 初始化checkpoint最后使用时间
last_access_time = time.time()
checkpoint_lock = Lock()
checkpoint = None


def clear_checkpoint():
    global last_access_time, checkpoint
    # 每10分钟检查，如果checkpoint闲置时间在10分钟以上将卸载它.闲置时间必须大于每次处理掩码的时间。最好设置高一点，不然在使用途中也可能被卸载。
    with checkpoint_lock:
        if time.time() - last_access_time > 60 * 10:
            checkpoint = None
            # 如果被载入到GPU专用内存，则需要使用torch来清除缓存。
            torch.cuda.empty_cache()
            # 需要显式地回收
            gc.collect()
            print("Checkpoint cleared due to inactivity.")


def load_checkpoint_if_needed():
    global checkpoint
    with checkpoint_lock:
        if checkpoint is None:
            # 载入checkpoint
            checkpoint = CHECKPOINT


def clear_temp():
    current_time = time.time()
    # 由请求生成的图片、掩码等临时文件的寿命
    max_age = 60 * 60

    for filename in os.listdir(UPLOAD):
        file_path = os.path.join(UPLOAD, filename)
        file_age = current_time - os.path.getctime(file_path)

        if file_age > max_age:
            os.remove(file_path)
            print(f"Deleted old file: {filename}")


# 每60分钟清理一次临时文件，2分钟检查checkpoint是否闲置在10分钟以上（闲置时间10~12分钟）
app.config['JOBS'] = [
    {
        'id': 'clear_temp',
        'func': clear_temp,
        'trigger': 'interval',
        'minutes': 60,
    },
    {
        'id': 'clear_checkpoint',
        'func': clear_checkpoint,
        'trigger': 'interval',
        'minutes': 2,  # 十分钟检查一次
    }
]

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()


# 上传图片生成掩码
@app.route('/upload_image', methods=['POST'])
def upload_image():
    # 确保checkpoints载入，因为这个函数内已经带checkpoint_lock锁了，不能再带一次。
    load_checkpoint_if_needed()
    # 需要带锁
    with checkpoint_lock:
        global last_access_time
        # 更新使用时间，用它来检查一段时间如果没有使用模型，就回收checkpoint
        last_access_time = time.time()

        unique_id = uuid.uuid4()
        session['unique_id'] = str(unique_id)
        file_name = f"{unique_id}.png"

        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        # 验证文件是否为图像
        if not allowed_file(file.filename):
            return "Invalid file type", 415  # 415 不支持的媒体格式（要求是图片）

        if file:
            save_path = os.path.join(app.config['UPLOAD'], file_name)
            file.save(save_path)
            input_image_path = f'./upload/{file_name}'
            masks_res = f'./upload/{unique_id}.txt'

            num_masks = generate_masks(input_image_path, checkpoint, masks_res)

            print(f"Generated {num_masks} masks.")

            del input_image_path, masks_res
            gc.collect()
            # 用完也要更新checkpoint使用时间
            last_access_time = time.time()
            return f"{num_masks} Masks generated and saved", 200
        else:
            return "Error in processing", 500


# 根据session获取掩码
@app.route('/result')
def result():
    upload_folder = UPLOAD
    unique_id = session['unique_id']
    # 使用session来确保用户只能访问自己的内容
    masks = f'{unique_id}.txt'
    return send_from_directory(upload_folder, masks)


@app.route('/draw_masks', methods=['GET'])
def generate_mask_image():
    unique_id = session['unique_id']
    image = f'./upload/{unique_id}.png'
    masks_txt = f'./upload/{unique_id}.txt'
    output_path = f'./upload/{unique_id}.png'  # 指定输出路径

    # 调用函数处理数据并保存图像
    draw_with_masks(image, masks_txt, output_path)

    # 确定目录和文件名
    dir = os.path.dirname(output_path)
    filename = os.path.basename(output_path)

    # 返回图像文件
    return send_from_directory(directory=dir, path=filename, mimetype='image/png')


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    # 如果 UPLOAD 目录存在，删除它及其所有内容
    if os.path.exists(UPLOAD):
        shutil.rmtree(UPLOAD)
    # 重新创建
    if not os.path.exists(UPLOAD):
        os.makedirs(UPLOAD)
    app.run(host='0.0.0.0', port=5000)
