rm -rf _build/html/*
rm -rf api/*
ls "api/"
sphinx-apidoc -E -e -M -o api/ ../threedb/
python reformat.py
make html
cp -r _build/html ../docs
