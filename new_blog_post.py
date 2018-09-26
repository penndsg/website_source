#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 Ben Lindsay <benjlindsay@gmail.com>

import sys
from datetime import datetime
from os.path import isfile

TEMPLATE = """---
layout: post
title: "{title}"
# author:
# image: /images/blog/{slug}.png
---

Blog post content here. To see what you can do with Markdown, check out
[this Markdown cheatsheet]
(https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).
"""


def new_blog_post(title):
    today = datetime.today()
    slug = title.lower().strip().replace(' ', '-')
    f_create = "blog/_posts/{}-{:0>2}-{:0>2}-{}.md".format(
        today.year, today.month, today.day, slug)
    t = TEMPLATE.strip().format(title=title, slug=slug)
    if isfile(f_create):
        print("{} already exists. ".format(f_create) +
              "You can either delete it or pick a different title.")
    else:
        with open(f_create, 'w') as w:
            w.write(t)
            w.write('\n')
        print("File created -> " + f_create)
        print("Open this file and add author name and (optionally) an image " +
              "at images/blog/{}.png.".format(slug))


if __name__ == '__main__':

    if len(sys.argv) > 1:
        new_blog_post(sys.argv[1])
    else:
        print(
            "No title given. Usage: {} \"Blog Post Title\"".format(sys.argv[0])
        )
