rm -rf _build/html/*
rm -rf api/*
ls "api/"
# sphinx-apidoc -e -M -o api/ ../threedb/
sphinx-apidoc -fMeT -o api/ ../threedb/
make html