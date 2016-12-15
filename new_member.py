#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 Ben Lindsay <benjlindsay@gmail.com>

import sys
from datetime import datetime

TEMPLATE = """---
layout: member
title: "{name}"
degree: "[Degree] [Program] [Grad Year]"
# cv: /pdfs/members/{slug}-cv.pdf
# resume: /pdfs/members/{slug}-resume.pdf
# twitter: your_twitter_handle
# github: your_github_handle
# bitbucket: your_bitbucket_handle
# scholar: your_google_scholar_id
# linkedin: your_linkedin_id (the part that goes after linkedin.com/in/)
# researchgate: your_researchgate_id
# website: yourwebsite.com
# email: name at example.edu
# phone: "(123) 456-7890"
# image: /images/members/{slug}.jpg
---

Tell us about yourself here. You can add links like [this](https://www.google.com/search?q=this).
"""


def new_member(name):
    today = datetime.today()
    slug = name.lower().strip().replace(' ', '-')
    f_create = "members/_posts/{}-{:0>2}-{:0>2}-{}.md".format(
        today.year, today.month, today.day, slug)
    t = TEMPLATE.strip().format(name=name, slug=slug)
    with open(f_create, 'w') as w:
        w.write(t)
        w.write('\n')
    print("File created -> " + f_create)
    print("Edit this file and add square profile picture at images/members/{}.jpg".format(slug))


if __name__ == '__main__':

    if len(sys.argv) > 1:
        new_member(sys.argv[1])
    else:
        print "No name given. Usage: {} \"Your Name\"".format(sys.argv[0])
