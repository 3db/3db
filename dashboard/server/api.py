from flask import Flask
from flask import Response
import numpy as np
from tqdm import tqdm
import json
from os import path
import argparse
import gzip, functools
from io import BytesIO as IO
from flask import after_this_request, request
from flask_cors import CORS

from flask import send_from_directory



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('logdir', type=str,
                    help='where to find the log information')



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
                    result = np.zeros((n_samples, len(self.keys) + 4), dtype='object')
                else:
                    assert self.keys == cur_keys

                for kix, k in enumerate(self.keys):
                    result[i, kix] = render_args[k]

                result[i, -1] = data['is_correct']
                result[i, -2] = data['environment']
                result[i, -3] = data['model']
                result[i, -4] = data['id']
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

    @app.route('/images/<imid>')
    def send_js(imid):
        return send_from_directory(path.join(args.logdir, 'images'), imid + '.png')


    @app.route('/')
    @gzipped
    def return_data():
        reader.update_data()
        return Response(reader.answer, mimetype='application/json')

    app.run(host='0.0.0.0', port='8000', debug=True)
