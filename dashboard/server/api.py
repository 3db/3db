import io
from flask import Flask
import cv2
from flask import Response, abort, send_file
import numpy as np
from tqdm import tqdm
import json
from os import path
import os
import sys
import argparse
import gzip, functools
from io import BytesIO as IO
from flask import after_this_request, request
from torch_utils import transform_image, get_prediction, render_prediction
from robustness import model_utils, datasets
from flask_cors import CORS

from flask import send_from_directory

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('logdir', type=str,
                    help='where to find the log information')
parser.add_argument('--arch', type=str, default='resnet18')

def is_safe_path(basedir, path, follow_symlinks=True):
  if follow_symlinks:
    return os.path.realpath(path).startswith(basedir)
  return os.path.abspath(path).startswith(basedir)

def compute_overlay(img):
    img = cv2.Canny(img, 200, 200)
    img *= img > 0
    alpha = (img * 0 + 255) * (img > 0)
    img = np.stack([img * 0, img* 0, img*1, alpha], 2)
    return img

class DataReader():

    def __init__(self, fname):
        self.fname = fname
        self.last_size = 0
        self.next_ix = 0
        self.keys = None
        self.result = None
        self.answer = '{}'

    def create_block(self, n):
        assert self.keys is not None
        return np.zeros((n, len(self.keys) + 4), dtype='object')

    def has_changed(self):
        current_size = path.getsize(self.fname)
        return current_size > self.last_size

    def update_data(self):
        # Make sure this is identical to EXTRA_KEYS in DetailView.js on the client
        EXTRA_KEYS = ['is_correct', 'environment', 'model', 'id', 'outputs']
        result = None
        with open(self.fname) as handle:
            full_data = handle.readlines()
            handle.seek(0, 2)
            self.last_size = handle.tell()
            n_samples = len(full_data) - self.next_ix - 1
            print("###", self.next_ix, len(full_data), n_samples)
            for i in tqdm(range(n_samples)):
                json_data = full_data[self.next_ix + i]
                data = json.loads(json_data)
                render_args = data['render_args']
                cur_keys = tuple(sorted(list(render_args.keys())))

                if self.keys is None:
                    self.keys = cur_keys

                if result is None:
                    result = np.zeros((n_samples, len(self.keys) + len(EXTRA_KEYS)), dtype='object')
                else:
                    assert self.keys == cur_keys

                for kix, k in enumerate(self.keys):
                    result[i, kix] = render_args[k]
                
                for kix, k in enumerate(EXTRA_KEYS):
                    result[i, -1-kix] = data[k]

            self.next_ix = len(full_data)
        if self.result is None:
            self.result = result
        else:
            if result is not None:
                print(self.result.shape, result.shape)
                self.result = np.concatenate([self.result, result])
        self.prepare_answer()

    def prepare_answer(self):
        self.answer = json.dumps({
            'parameters': list(self.keys),
            'data': self.result.tolist()
        })

def gzipped(f):
    @functools.wraps(f)
    def view_func(*args, **kwargs):
        @after_this_request
        def zipper(response):
            accept_encoding = request.headers.get('Accept-Encoding', '')

            if 'gzip' not in accept_encoding.lower():
                return response

            response.direct_passthrough = False

            if (response.status_code < 200 or
                response.status_code >= 300 or
                'Content-Encoding' in response.headers):
                return response
            gzip_buffer = IO()
            gzip_file = gzip.GzipFile(mode='wb',
                                      fileobj=gzip_buffer)
            gzip_file.write(response.data)
            gzip_file.close()

            response.data = gzip_buffer.getvalue()
            response.headers['Content-Encoding'] = 'gzip'
            response.headers['Vary'] = 'Accept-Encoding'
            response.headers['Content-Length'] = len(response.data)

            return response

        return f(*args, **kwargs)

    return view_func

if __name__ == '__main__':

    args = parser.parse_args()
    app = Flask(__name__,)
    CORS(app)

    print("STARTING")
    reader = DataReader(path.join(args.logdir, 'details.log'))

    print("Loading model...")
    ds = datasets.ImageNet("")
    model, _ = model_utils.make_and_restore_model(arch=args.arch, dataset=ds, pytorch_pretrained=True)
    model = model.model.eval().cuda()

    print("Loading class map...")
    img_class_map = None
    mapping_file_path = 'index_to_name.json'                  # Human-readable names for Imagenet classes
    if os.path.isfile(mapping_file_path):
        with open (mapping_file_path) as f:
            img_class_map = json.load(f)

    @app.route('/canny/<imid>')
    def send_canny(imid):
        full_path = path.join(args.logdir, 'images', imid + '.png')
        if is_safe_path(args.logdir, full_path):
            img = cv2.imread(full_path)
            canny = compute_overlay(img)
            is_success, buffer = cv2.imencode(".png", canny)
            io_buf = io.BytesIO(buffer)
            if not is_success:
                return abort(500)
            return send_file(io_buf, mimetype='image/png')

        else:
            return abort(400)

    @app.route('/images/<imid>')
    def send_js(imid):
        return send_from_directory(path.join(args.logdir, 'images'), imid + '_rgb.png')

    @app.route('/predict', methods=['POST'])
    def predict():
        if request.method == 'POST':
            f = request.files['file']
            if f is not None:
                input_tensor = transform_image(f).cuda()
                prediction_idx = get_prediction(model, input_tensor)
                class_id, class_name = render_prediction(prediction_idx, img_class_map)
                return jsonify({'class_id': class_id, 'class_name': class_name})
                
    @app.route('/')
    @gzipped
    def return_data():
        reader.update_data()
        return Response(reader.answer, mimetype='application/json')

    app.run(host='0.0.0.0', port='8001', debug=True)
