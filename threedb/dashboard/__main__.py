"""
threedb.dashboard
=================

This Module provides and lauch an API that the dashboard client can
connect to in order to pull the relevant data
"""

import argparse
from os import path

import webbrowser
from flask import Flask, Response
from flask_cors import CORS
from flask import send_from_directory
from flask_compress import Compress
from .data_reader import DataReader

import threedb

THREEDB_FOLDER = path.abspath(path.dirname(threedb.__file__))
DASHBOARD_UI = path.join(THREEDB_FOLDER, 'dashboard_html')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='3DB dashboard API')
    parser.add_argument('--port', '-p', type=int, default=8001,
                        help="The port this api will serve on")

    parser.add_argument('--no-browser', '-n', action='store_true',
                        help='Do not spawn a browser with the front-end')
    parser.add_argument('logdir', type=str,
                        help='where to find the log information')

    args = parser.parse_args()
    app = Flask(__name__,)
    compress = Compress()
    compress.init_app(app)
    CORS(app)

    reader = DataReader(args.logdir)
    reader.update_data()

    @app.route('/images/<imid>')
    def send_js(imid):
        return send_from_directory(path.join(path.realpath(args.logdir), 'images'), imid + '_rgb.png')

    @app.route('/log.json')
    def return_data():
        reader.update_data()
        return Response(reader.answer, mimetype='application/json')

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def catch_all(path):
        print("PATH", path)
        if not path:
            path = 'index.html'
        return send_from_directory(DASHBOARD_UI, path)

    if not args.no_browser:
        webbrowser.open(f'http://localhost:{args.port}?url=localhost:{args.port}', new=2)
        pass
    app.run(host='0.0.0.0', port=args.port, debug=False)
