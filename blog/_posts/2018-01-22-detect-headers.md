---
layout: post
title: Detect Headers in CSV Tables Statistically
author: Mingyang Li
---

Generally speaking, a CSV table may or may not contain a header, and the header, when it exists, may consists of multiple lines. Question: How can we detect which first rows constitutes the header of a CSV file?

I wrote a script, named headsman, that finds <b>the best amount of first few rows to cut off from a table</b> in order to maximize the purity of datatypes in each column.

What do I mean?

Notice that, in a column of a table, even if it contains only numerical entries, its header (if any) is often a string. For example, the table

<img class="progressiveMedia-image js-progressiveMedia-image" data-src="https://cdn-images-1.medium.com/max/1600/1*2qe35S0KsMM00eZDzOL0Gw.png" src="https://cdn-images-1.medium.com/max/1600/1*2qe35S0KsMM00eZDzOL0Gw.png">

contains the following datatypes in each cell

<img class="progressiveMedia-image js-progressiveMedia-image" data-src="https://cdn-images-1.medium.com/max/1600/1*vaSkczSazfDMtqmG40xoDw.png" src="https://cdn-images-1.medium.com/max/1600/1*vaSkczSazfDMtqmG40xoDw.png">

so the purity in each column is
<ul>
<li>The first column: 100%</li>
<li>The second column: 67%</li>
</ul>
whose average is 83%. However, if we cut off the first row, the first and the second columns will only contain strings and numbers, respectively.

<img class="graf-image" data-image-id="1*HkFmJyY1MyWhuFTaVCkpuw.png" data-width="801" data-height="79" data-action="zoom" data-action-value="1*HkFmJyY1MyWhuFTaVCkpuw.png" src="https://cdn-images-1.medium.com/max/1600/1*HkFmJyY1MyWhuFTaVCkpuw.png">

This gives purities of
<ul>
<li>The first column: 100%</li>
<li>The second column: 100%</li>
</ul>
the average of which is 100%, an increase over the former value of 67%. Hence, at least for tables with numerical-valued column(s), we can define headers as:

<b>Apparent header:</b> the first few rows whose removal would maximize the column-averaged purity in datatypes.

To make the argument more obvious, I used standard deviation weighted by number of rows remaining, averaged across all columns in each table, and standardized to 0~1. I call this metric “Data Type Deviation” (DTD).

Technical details:

<b>Standard deviation</b> tells me how heterogeneous the remaining cells in each column is.
They are <b>weighted by number of rows remaining,</b> so that unnecessary cut-offs will lower these standard deviations. These weights convert saturation points into turning points, as we will see in the plot below.
Weighted standard deviations (WSDs) of all columns in each table is <b>averaged and</b> scaled so that it’s maximum value is 1. Since the aforementioned weight, i.e. number of remaining rows, will always be zero when we have cut off all rows in each table, by scaling the max value to 1, I am essentially <b>standardizing</b> the WSDs (i.e., scaling the series of values so that their max is 1 and their min is 0).
I ran this metric over some thousands of tables derived from <a href="http://wrds-web.wharton.upenn.edu/" target="_blank">Wharton Research Data Services</a> (a big shout-out). The results looks like:

<img class="progressiveMedia-image js-progressiveMedia-image" data-src="https://cdn-images-1.medium.com/max/1200/1*sOFLQLtKpAp-kDFSpLUNEA.png" src="https://cdn-images-1.medium.com/max/1200/1*sOFLQLtKpAp-kDFSpLUNEA.png">

Each curve represents a table. Notice how each curve has a turning point achieving the y-value of 1. These are the turning points representing the last rows of headers.

A quick histogram tells us that most tables does not have an apparent header:

<img class="progressiveMedia-image js-progressiveMedia-image" data-src="https://cdn-images-1.medium.com/max/1600/1*vuc2isw85ykQ4jik28g6iQ.png" src="https://cdn-images-1.medium.com/max/1600/1*vuc2isw85ykQ4jik28g6iQ.png">

Notice how the first bin dominated the scale.

<h4><b>Efficiency Improvement</b></h4>
Intuitively, the header should not contain too many rows. Eliminating the tables who does not have an apparent header, the following histogram fits our expectation:

<img class="progressiveMedia-image js-progressiveMedia-image" data-src="https://cdn-images-1.medium.com/max/1600/1*SB4bCtOFC92n6QG3vYuJwg.png" src="https://cdn-images-1.medium.com/max/1600/1*SB4bCtOFC92n6QG3vYuJwg.png">

The fifth row (if present — we may have tables with 4 or less rows!) seems to be a good position to stop header detection. This greatly improved Headsman’s efficiency.
