#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 Ben Lindsay <benjlindsay@gmail.com>

import sys
from datetime import datetime
from os.path import isfile

TEMPLATE = """---
layout: member
title: "{name}"
degree: "{deg_string}"
# cv: /pdfs/members/{name_slug}-cv.pdf
# resume: /pdfs/members/{name_slug}-resume.pdf
# twitter: your_twitter_handle
# github: your_github_handle
# bitbucket: your_bitbucket_handle
# scholar: your_google_scholar_id
# linkedin: your_linkedin_id (the part that goes after linkedin.com/in/)
# researchgate: your_researchgate_id
# website: yourwebsite.com
# email: {email}
# phone: "(123) 456-7890"
# image: /images/members/{name_slug}.jpg
---
"""


def new_member(name, deg_string='', date_slug=None, email=None):
    if date_slug is None:
        today = datetime.today()
        date_slug = '{}-{:0>2}-{:0>2}'.format(today.year, today.month, today.day)
    if email is None:
        email = 'name at example.com'
    name_slug = name.lower().strip().replace(' ', '-')
    f_create = "members/_posts/{}-{}.md".format(date_slug, name_slug)
    t = TEMPLATE.strip().format(name=name, deg_string=deg_string,
                                name_slug=name_slug, email=email)
    if isfile(f_create):
        print("{} already exists. ".format(f_create) +
              "You can either delete it or pick a different title.")
    else:
        with open(f_create, 'w') as w:
            w.write(t)
            w.write('\n')
        print("File created -> " + f_create)
        print("Edit this file and add square profile picture at " +
              "images/members/{}.jpg".format(name_slug))


if __name__ == '__main__':

    if len(sys.argv) > 1:
        new_member(sys.argv[1])
    else:
        print("No name given. Usage: {} \"Your Name\"".format(sys.argv[0]))
