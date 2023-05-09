<div align="center"><img src=pic/icon/BC_Title.png><img src=pic/icon/Logo.png width="180" /></div>
<div align=center>
  
“𝑇ℎ𝑒 𝑏𝑎𝑏𝑦, 𝑎𝑠𝑠𝑎𝑖𝑙𝑒𝑑 𝑏𝑦 𝑒𝑦𝑒𝑠, 𝑒𝑎𝑟𝑠, 𝑛𝑜𝑠𝑒, 𝑠𝑘𝑖𝑛, 𝑎𝑛𝑑 𝑒𝑛𝑡𝑟𝑎𝑖𝑙𝑠 𝑎𝑡 𝑜𝑛𝑐𝑒, 𝑓𝑒𝑒𝑙𝑠 𝑖𝑡 𝑎𝑙𝑙 𝑎𝑠 𝑜𝑛𝑒 𝑔𝑟𝑒𝑎𝑡 𝑏𝑙𝑜𝑜𝑚𝑖𝑛𝑔, 𝑏𝑢𝑧𝑧𝑖𝑛𝑔 𝑐𝑜𝑛𝑓𝑢𝑠𝑖𝑜𝑛.” -- 𝑊𝑖𝑙𝑙𝑖𝑎𝑚 𝐽𝑎𝑚𝑒𝑠
  
</div>

# <span id="head-content"> *🍓 Content 🍓* </span>
* - [ ] [**1. Introduction**](#head1)
* - [ ] [**2. Background**](#head2)
  * - [ ] [2.1 Datasets](#head-dataset)
  * - [ ] [2.2 Evaluation Metrics](#head-metrics)
* - [ ] [**3. Generative Models**](#head3)
  * - [ ] [3.1 GAN-based Model](#head-gan)
  * - [ ] [3.2 Transformer-based Model](#head-transformer)
  * - [ ] [3.3 Diffusion-based Model](#head-diffusion)
* - [ ] [**4. Generative Applications**](#head4)
  * - [ ] [4.1 Text-to-Image](#head-T2I)
  * - [ ] [4.1 Text-to-X](#head-T2X)
  * - [ ] [4.1 X-to-Image](#head-X2I)
  * - [ ] [4.4 Multi Tasks](#head-multi)
* - [ ] [**5. Discussion**](#head5)
  * - [ ] [5.1 Compounding Issues](#head-issue)
    * - [ ] [Computational Aesthetic](#head-aesthetic)
    * - [ ] [Prompt Engineering](#head-prompt)
  * - [ ] [5.2 Business Analysis](#head-business)
    * - [ ] [Online Platforms](#head-online)
    * - [ ] [Ethical Considerations](#head-prompt)
  * - [ ] [5.3 Challenges & Future Outlooks](#head-future)

## <span id="head1"> *1. Introduction* </span> [       «🎯Back To Top»       ](#)
The human perceptual system is a complex and multifaceted construct. The five basic senses of **hearing, touch, taste, smell, and vision** serve as primary channels of perception, allowing us to perceive and interpret most of the external stimuli encountered in this “blooming, buzzing confusion” world. These stimuli always come from **multiple events** spread out spatially and temporally distributed. 

In other words, we constantly perceive the world in a “**multimodal**” manner, which combines different information channels to distinguish features within confusion, seamlessly integrates various sensations from multiple modalities and obtains knowledge through our experiences. 

## <span id="head2"> *2. Background* </span> [       «🎯Back To Top»       ](#)
### <span id="head-dataset"> *2.1 Datasets* </span> [       «🎯Back To Top»       ](#)

Table 1. **Chronological timeline of representative text-to-image datasets.** 

>*“Public”* includes a link to each dataset (if available✔) or paper (if not❌). *“Annotations”* denotes the number of text descriptions per image. *“Attrs”* denotes the total number of attributes in each dataset.

| Year | Dataset | Public | *Category* | *Image (Resolution)* | *Annotations* | *Attrs* | *Other Information* |
| -------- | :-------- |  :------------: | :------------: | :------------: | :------------: | :------------: | :-------- |
| 2008 |  Oxford-102 Flowers | [✔](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) | Flower | 8,189 (-) | 10 | - | - |
| 2011 |  CUB-200-2011 | [✔](http://www.vision.caltech.edu/datasets/cub_200_2011/) | Bird | 11,788 (-) | 10 | - | BBox, Segmentation... |
| 2014 |  MS-COCO2014 |  [✔](https://cocodataset.org/#overview)|  Iconic Objects |  120k (-)|  5 |  - |  BBox,Segmentation... |  
| 2018 |  Face2Text |  [✔](https://github.com/mtanti/face2text-dataset/) |  Face |  10,177 (-)|  1~ |  -|  -|  
| 2019 |  SCU-Text2face |  [❌](https://arxiv.org/abs/1904.05729) |  Face |  1,000 (256×256)|  5 |  -|  -|  
| 2020 |  Multi-ModalCelebA-HQ |  [✔](https://github.com/IIGROUP/MM-CelebA-HQ-Dataset)|  Face |  30,000 (512×512)|  10 |  38 |  Masks,Sketches |  
| 2021 |  FFHQ-Text |  [✔](https://github.com/Yutong-Zhou-cv/FFHQ-Text_Dataset)|  Face |  760 (1024×1024)|  9 |  162 |  BBox|  
| 2021 |  M2C-Fashion |  [❌](https://proceedings.neurips.cc/paper/2021/hash/e46bc064f8e92ac2c404b9871b2a4ef2-Abstract.htm)|  Clothing |  10,855,753 (256×256)|  1|  -|  -|  
| 2021 |  CelebA-Dialog |  [✔](http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Dialog.html)|  Face|  202,599 (178×218)|  ~5 |  5|  Identity Label...|  
| 2021 |  Faces a la Carte |  [❌](https://openaccess.thecvf.com/content/WACV2021/papers/Wang_Faces_a_la_Carte_Text-to-Face_Generation_via_Attribute_Disentanglement_WACV_2021_paper.pdf)|  Face|  202,599 (178×218)|  ~10|  40|  -|  
| 2021 |  LAION-400M |  [✔](https://laion.ai/blog/laion-400-open-dataset/)|  Random Crawled|  400M (-)|  1|  -|  KNN Index...|  
| 2022 |  Bento800 |  [✔](https://github.com/Yutong-Zhou-cv/Bento800_Dataset)|  Food|  800 (600×600)|  9|  -|  BBox, Segmentation, Label...|  
| 2022 |  LAION-5B |  [✔](https://laion.ai/blog/laion-5b/)|  Random Crawled|  5.85B (-)|  1|  -|  URL, Similarity, Language...|  
| 2022 |  DiffusionDB |  [✔](https://poloclub.github.io/diffusiondb/)|  Synthetic Images|  14M (-)|  1|  -|  Size, Random Seed...|  
| 2022 |  COYO-700M |  [✔](https://github.com/kakaobrain/coyo-dataset)|  Random Crawled|  747M (-)|  1|  -|  URL, Aesthetic Score...|  
| 2022 |  DeepFashion-MultiModal |  [✔](https://github.com/yumingj/DeepFashion-MultiModal)|  Full Body|  44,096 (750×1101)|  1|  -|  Densepose, Keypoints...|  
| 2023 |  ANNA |  [✔](https://github.com/aashish2000/ANNA)|  News |  29,625 (256×256)|  1 |  -|  -|  
| 2023 |  DreamBooth |  [✔](https://github.com/google/dreambooth)|  Objects & Pets|  158 (-)|  25 |  -|  -|  

### <span id="head-metrics"> *2.2 Evaluation Metrics* </span> [       «🎯Back To Top»       ](#)
* **Automatic Evaluation**

* **Human Evaluation**

## <span id="head3"> *3. Generative Models* </span> [       «🎯Back To Top»       ](#)
 
## <span id="head4"> *4. Generative Applications* </span> [       «🎯Back To Top»       ](#)

## <span id="head5"> *5. Discussion* </span> [       «🎯Back To Top»       ](#)
