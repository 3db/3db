#!/bin/bash
REPO=$1

rm -rf ./dashboard_source
git clone $REPO ./dashboard_source
cd ./dashboard_source
npm install
npm run build
npm run export
rm -rf ../threedb/dashboard_html
mv out ../threedb/dashboard_html
cd ..
rm -rf ./dashboard_source
