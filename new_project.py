#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 Ben Lindsay <benjlindsay@gmail.com>

import sys
from datetime import datetime

TEMPLATE = """---
layout: project
title: "{title}"
current: true
contributors:
 - Team Leader Name
 - Team Member 2
 - Team Member 3
# image: /images/projects/{slug}.png
---

Project description here. You can add links like [this](https://www.google.com/search?q=this).
"""


def new_member(title):
    today = datetime.today()
    slug = title.lower().strip().replace(' ', '-')
    f_create = "projects/_posts/{}-{:0>2}-{:0>2}-{}.md".format(
        today.year, today.month, today.day, slug)
    t = TEMPLATE.strip().format(title=title, slug=slug)
    with open(f_create, 'w') as w:
        w.write(t)
        w.write('\n')
    print("File created -> " + f_create)
    print("Open this file and add contributor names and project description")


if __name__ == '__main__':

    if len(sys.argv) > 1:
        new_member(sys.argv[1])
    else:
        print("No title given. Usage: {} \"Project Title\"".format(sys.argv[0]))
