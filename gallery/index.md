---
layout: default
title: Gallery
big_header: true
categories:
 - gallery

images:
  #- image_path: DSC_0806.jpg
  #- image_path: DSC_0808.jpg
  #- image_path: DSC_0816.jpg
  #- image_path: DSC_0820.jpg
  #- image_path: DSC_0835.jpg
  #- image_path: DSC_0842.jpg
  #- image_path: DSC_0845.jpg
  #- image_path: DSC_0880.jpg
  #- image_path: DSC_1347.jpg
  #- image_path: DSC_1353.jpg
  #- image_path: DSC_1362.jpg
  - image_path: IMG_20170914_182400.jpg
  - image_path: IMG_20170914_182405.jpg
  - image_path: IMG_20170914_182408.jpg
  - image_path: 20170126-IMG_9112.jpg
  - image_path: 20170126-IMG_9131.jpg
  - image_path: 20170126-IMG_9135.jpg
  - image_path: 20170126-IMG_9141.jpg
  - image_path: 20170126-IMG_9143.jpg
  - image_path: 20170126-IMG_9150.jpg
  - image_path: 20170126-IMG_9161.jpg
  - image_path: 20170126-IMG_9167.jpg
  - image_path: 20170126-IMG_9168.jpg
  - image_path: Healthcare-1.jpg
  - image_path: Healthcare-2.jpg
  - image_path: Healthcare-3.jpg
  - image_path: Healthcare-4.jpg
---

<ul id="lightSlider">
  {% for image in page.images %}
    <li><img size="50%" src="{{ image.image_path }}" alt="PDSG Event"/></li>
  {% endfor %}
</ul>
<script type="text/javascript">
    $(document).ready(function() {
      $("#lightSlider").lightSlider(); 
    });
</script>

