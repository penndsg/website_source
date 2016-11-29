#!/bin/bash
#
# deploy.sh
#
# Copyright (c) 2016 Ben Lindsay <benjlindsay@gmail.com>

jekyll build

cd _site

if [ ! -d .git ]; then
  git init
  git remote add origin https://github.com/penndsg/penndsg.github.io.git
fi

git add .
git commit -m "Site updated on $(date)"
git push origin master
