import requests
import sys
from io import StringIO
from time import sleep
from copy import copy
from tqdm import tqdm
import pandas as pd

TIME_IN_BETWEEN = 5

STORE = []
KEYS_TO_KEEP = set(['price', 'inStore', 'reviewCount', 'animationCount',
                    'embedUrl', 'uid', 'likeCount', 'averageRating',
                    'viewCount', 'viewerUrl', 'name'])

def process_results(results, clazz):
    for result in results:
        processed_result = {k: v for (k, v) in result.items() if k in KEYS_TO_KEEP}
        processed_result['class'] = clazz
        all_images = sorted((result['thumbnails']['images']), key=lambda x: x['size'])
        for i, im_descriptor in enumerate(all_images):
            image_key = f"""image-{i}"""
            processed_result[image_key] = im_descriptor["url"]
        STORE.append(processed_result)



SEARCH_PARAMS = {
    'downloadable': 1,
    'sort_by': '-pertinence',
    'type': 'models',
}

def search_for_class(query):
    global TIME_IN_BETWEEN
    params = copy(SEARCH_PARAMS)
    params['q'] = query
    while True:
        print("hoho", file=sys.stderr)
        r = requests.get('https://sketchfab.com/i/search', params=params)
        code = r.status_code
        print(code, file=sys.stderr)
        if code == 429:
            TIME_IN_BETWEEN *= 2
        else:
            result_json = r.json()
            try:
                process_results(result_json['results'], query)
                next_cursor = result_json['cursors']['next']
                if next_cursor is None:
                    break
                params['cursor'] = next_cursor
            except:
                break
        sleep(TIME_IN_BETWEEN)


if __name__ == '__main__':
    classes = sys.stdin.readlines()

    for clazz in tqdm(classes):
        clazz = clazz.strip()
        search_for_class(clazz)

    frame = pd.DataFrame.from_records(STORE)
    output = StringIO()
    frame.to_csv(output)
    output.seek(0)
    print(output.read())
