from glob import glob
from pathlib import Path
import os

all_api_docs = glob('api/*.rst')
for api_doc in all_api_docs:
    lines = open(api_doc).readlines()
    delete_next = 0
    new_lines = []
    for line in lines:
        if delete_next > 0:
            delete_next -= 1
        elif ('Subpackages' in line) or ('Submodules' in line):
            print('Stripping title from', api_doc)
            delete_next += 1
        else:
            new_lines.append(line)
    open(api_doc, 'w').writelines(new_lines)

all_image_dirs = glob('_static/logs/*/images')
for image_dir in map(Path, all_image_dirs):
    contents = os.listdir(image_dir)
    if not all([x.startswith('image') for x in contents]):
        for i, image_name in enumerate(sorted(contents)):
            print(f'Renaming {image_name} -> image_{i+1}.png')
            os.rename(image_dir / image_name, image_dir / f'image_{i+1}.png')