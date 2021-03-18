rm -rf _build/html/*
rm -rf api/*
ls "api/"
sphinx-apidoc -e -M -o api/ ../threedb/
make html
