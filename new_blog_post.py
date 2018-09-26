#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 Ben Lindsay <benjlindsay@gmail.com>

import sys
from datetime import datetime

TEMPLATE = """---
layout: post
title: "{title}"
# author:
# image: /images/blog/{slug}.png
---

Project description here. You can add links like [this](https://www.google.com/search?q=this).
"""


def new_blog_post(title):
    today = datetime.today()
    slug = title.lower().strip().replace(' ', '-')
    f_create = "blog/_posts/{}-{:0>2}-{:0>2}-{}.md".format(
        today.year, today.month, today.day, slug)
    t = TEMPLATE.strip().format(title=title, slug=slug)
    with open(f_create, 'w') as w:
        w.write(t)
        w.write('\n')
    print("File created -> " + f_create)
    print("Open this file and add author name and (optionally) an image at " +
          "images/blog/{}.png.".format(slug))


if __name__ == '__main__':

    if len(sys.argv) > 1:
        new_blog_post(sys.argv[1])
    else:
        print(
            "No title given. Usage: {} \"Blog Post Title\"".format(sys.argv[0])
        )
