import random
import requests
import zipfile
import sys
import pickle
from os import path
from io import StringIO, BytesIO
from time import sleep
from copy import copy
from tqdm import tqdm
import pandas as pd
import string


def randomString(stringLength=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

TIME_IN_BETWEEN = 5
CLASS_FILE = './classes.txt'
LABELING_FOLDER = './labeling'
OUTPUT_FOLDER = '/data2/Microsoft/OSimModels'

CREATE_ACCOUNT_URL = "https://sketchfab.com/i/users"
CSRF_URL = "https://sketchfab.com/i/csrf"
LOGIN_URL = "https://sketchfab.com/login"
SETTINGS_URL = "https://sketchfab.com/settings/password"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36"


def create_account():

    username = randomString(10)
    email = f"""{username}@gmail.com"""
    password = randomString(15)
    print(username, password)

    query = requests.post(CREATE_ACCOUNT_URL, data={
        'username': username,
        'email': email,
        'password': password,
        'gdpr_consent': True,
        'recaptcha': randomString(5)
    })

    if query.status_code == 201:
        return username, email, password
    else:
        raise Exception(query.text)

def login(email, password):
    session = requests.Session()
    response = requests.get(CSRF_URL)
    cookies =dict(response.cookies)
    r = session.post(LOGIN_URL, params={'DEBUG': True}, headers={
        'referer': 'https://sketchfab.com/?logged_out=1',
        'x-csrftoken': cookies['sb_csrftoken'],
        'x-requested-with': 'XMLHttpRequest'
    }, cookies=cookies, data={
        'email': email,
        'password': password,
        'recaptcha': randomString(5),
        'next': '/feed'
    })
    return session

def get_api_token(session):
    result = session.get(SETTINGS_URL, headers={
        'user-agent': USER_AGENT
    })
    lines = result.text.split('\n')
    for l in lines:
        if 'id="apitoken"' in l:
            token = l.split('value="')[-1].split('"')[0]
            return token
    raise Exception('No token found')

class API():

    def __init__(self):
        self.init_credentials()

    def init_credentials(self):
        _, email, password = create_account()
        session = login(email, password)
        self.api_token = get_api_token(session)
        print("# new token: ", self.api_token)

    def get(self, url, headers={}, **kwargs):
        while True:
            headers = copy(headers)
            headers['authorization'] = f"""Token {self.api_token}"""
            result = requests.get(url, headers=headers, **kwargs)
            if result.status_code != 429:
                return result
            self.init_credentials()  # Recreate an account when we are banned





def download_model(uid, folder, api):
    url = f"""https://api.sketchfab.com/v3/models/{uid}/download"""
    destination_path = path.join(folder, uid)
    if path.exists(destination_path):  # Do not redownload
        return
    request = api.get(url)
    if request.status_code != 403:
        json = request.json()
        try:
            aws_url = json['gltf']['url']
        except:
            print('GLTF not available', uid)
            return
        aws_request = requests.get(aws_url)
        zipdata = BytesIO()
        zipdata.write(aws_request.content)
        content = zipfile.ZipFile(zipdata)
        content.extractall(destination_path)
    sleep(TIME_IN_BETWEEN)


def get_all_classes(class_file):
    with open(class_file) as cl_file:
        classes = [x.strip().lower() for x in cl_file.readlines()]
        return classes

def get_selected_for_class(clazz, labeling_folder):
    target_path = path.join(labeling_folder, f"""class-{clazz}.pkl""")

    try:
        with open(target_path, 'rb') as h:
            store = pickle.load(h)
    except:
        store = {}

    return [k for (k, v) in store.items() if v]

if __name__ == '__main__':
    api = API()
    all_classes = get_all_classes(CLASS_FILE)
    selected_uids = []
    for clazz in all_classes:
        selected_uids.extend(get_selected_for_class(clazz, LABELING_FOLDER))

    for uid in tqdm(selected_uids):
        download_model(uid, OUTPUT_FOLDER, api)
