<div align="center"><img src=pic/icon/BC_Title.png><img src=pic/icon/Logo.png width="180" /></div>
<div align=center>
  
“𝑇ℎ𝑒 𝑏𝑎𝑏𝑦, 𝑎𝑠𝑠𝑎𝑖𝑙𝑒𝑑 𝑏𝑦 𝑒𝑦𝑒𝑠, 𝑒𝑎𝑟𝑠, 𝑛𝑜𝑠𝑒, 𝑠𝑘𝑖𝑛, 𝑎𝑛𝑑 𝑒𝑛𝑡𝑟𝑎𝑖𝑙𝑠 𝑎𝑡 𝑜𝑛𝑐𝑒, 𝑓𝑒𝑒𝑙𝑠 𝑖𝑡 𝑎𝑙𝑙 𝑎𝑠 𝑜𝑛𝑒 𝑔𝑟𝑒𝑎𝑡 𝑏𝑙𝑜𝑜𝑚𝑖𝑛𝑔, 𝑏𝑢𝑧𝑧𝑖𝑛𝑔 𝑐𝑜𝑛𝑓𝑢𝑠𝑖𝑜𝑛.” -- 𝑊𝑖𝑙𝑙𝑖𝑎𝑚 𝐽𝑎𝑚𝑒𝑠
  
</div>

# <span id="head-content"> *🍓 Content 🍓* </span>
* - [x] [**1. Introduction**](#head-1)
* - [x] [**2. Background**](#head-2)
  * - [x] [2.1 Datasets](#head-dataset)
  * - [x] [2.2 Evaluation Metrics](#head-metrics)
* - [x] [**3. Generative Models**](#head-3)
  * - [x] [3.1 GAN Model](#head-gan)
  * - [x] [3.2 Autogressive Model](#head-transformer)
  * - [x] [3.3 Diffusion Model](#head-diffusion)
* - [ ] [**4. Generative Applications**](#head-4)
  * - [x] [4.1 Text-to-Image](#head-T2I)
  * - [ ] [4.2 Text-to-X](#head-T2X)
  * - [ ] [4.3 X-to-Image](#head-X2I)
  * - [ ] [4.4 Multi Tasks](#head-multi)
* - [ ] [**5. Discussion**](#head-5)
  * - [ ] [5.1 Compounding Issues](#head-issue)
    * - [ ] [Computational Aesthetic](#head-aesthetic)
    * - [ ] [Prompt Engineering](#head-prompt)
  * - [ ] [5.2 Business Analysis](#head-business)
    * - [ ] [Online Platforms](#head-online)
    * - [ ] [Ethical Considerations](#head-prompt)
  * - [ ] [5.3 Challenges & Future Outlooks](#head-future)

## <span id="head-1"> *1. Introduction* </span> [       «🎯Back To Top»       ](#)
The human perceptual system is a complex and multifaceted construct. The five basic senses of **hearing, touch, taste, smell, and vision** serve as primary channels of perception, allowing us to perceive and interpret most of the external stimuli encountered in this “blooming, buzzing confusion” world. These stimuli always come from **multiple events** spread out spatially and temporally distributed. 

In other words, we constantly perceive the world in a “**multimodal**” manner, which combines different information channels to distinguish features within confusion, seamlessly integrates various sensations from multiple modalities and obtains knowledge through our experiences. 

## <span id="head-2"> *2. Background* </span> [       «🎯Back To Top»       ](#)
### <span id="head-dataset"> *2.1 Datasets* </span> [       «🎯Back To Top»       ](#)

Table 1. **Chronological timeline of representative text-to-image datasets.** 

>**“Public”** includes a link to each dataset (if available✔) or paper (if not❌).   
>***“Annotations”*** denotes the number of text descriptions per image.   
>***“Attrs”*** denotes the total number of attributes in each dataset.

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
> 👆🏻: **Higher** is better. 👇🏻: **Lower** is better.

   * [NIPS 2016] **Inception Score** (IS) 👆🏻
       * [[Paper](https://papers.nips.cc/paper_files/paper/2016/hash/8a3363abe792db2d8761d6403605aeb7-Abstract.html)] [[Python Code (Pytorch)](https://github.com/sbarratt/inception-score-pytorch)] [(New!)[Python Code (Tensorflow)](https://github.com/senmaoy/Inception-Score-FID-on-CUB-and-OXford)] [[Ref.Code(AttnGAN)](https://github.com/taoxugit/AttnGAN)]
   * [NIPS 2017] **Fréchet Inception Distance** (FID) 👇🏻
       * [[Paper](https://papers.nips.cc/paper_files/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html)] [[Python Code (Pytorch)](https://github.com/mseitzer/pytorch-fid)] [(New!)[Python Code (Tensorflow)](https://github.com/senmaoy/Inception-Score-FID-on-CUB-and-OXford)] [[Ref.Code(DM-GAN)](https://github.com/MinfengZhu/DM-GAN)]
   * [CVPR 2018)] **R-precision** (RP) 👆🏻
       * [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf)] [[Ref.Code(CPGAN)](https://github.com/dongdongdong666/CPGAN)]
   * [TPAMI 2020] **Semantic Object Accuracy** (SOA) 👆🏻
       * [[Paper](https://ieeexplore.ieee.org/document/9184960)] [[Python Code (Pytorch)](https://github.com/tohinz/semantic-object-accuracy-for-generative-text-to-image-synthesis/tree/master/SOA)] 
   * [ECCV 2022] **Positional Alignment** (PA) 👆🏻
       * [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-20059-5_34)] [[Python Code (Pytorch)](https://github.com/VinAIResearch/tise-toolbox/tree/master/positional_alignment)] 

* **Human Evaluation**
>Participants are asked to rate generated images based on two criteria: **plausibility** (including object accuracy, counting, positional alignment, or image-text alignment) and **naturalness** (whether the image appears natural or realistic).  
The evaluation protocol is designed in a **5-Point** Likert manner, in which human evaluators rate each prompt on a scale of 1 to 5, with **5 representing the best** and **1 representing the worst**. 

For **rare object combinations** that require common sense understanding or aim to **avoid bias related to race or gender**, human evaluation is even more important. 

## <span id="head-3"> *3. Generative Models* </span> [       «🎯Back To Top»       ](#)

<div align=center> 
  
**A comprehensive list of `text-to-image` approaches.** 

</div>

>The pioneering works in each development stage are `highlighted`. Text-to-face generation works are start with a emoji(👸).

### <span id="head-gan"> 3.1 **GAN** Model </span> [       «🎯Back To Top»       ](#)
  * **Conditional GAN-based**
    * 2016~2021: 
      * `Generative Adversarial Text to Image Synthesis` [[Paper](http://proceedings.mlr.press/v48/reed16.pdf)] [[Code](https://github.com/reedscot/icml2016)]
      * Learning What and Where to Draw [[Paper](https://arxiv.org/pdf/1610.02454.pdf)] [[Code](https://github.com/reedscot/nips2016)]
      * Adversarial nets with perceptual losses for text-to-image synthesis [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8168140)]
      * I2T2I: Learning Text to Image Synthesis with Textual Data Augmentation [[Paper](https://arxiv.org/pdf/1703.06676.pdf)] [[Code](https://github.com/zsdonghao/im2txt2im)]
      * Inferring Semantic Layout for Hierarchical Text-to-Image Synthesis [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hong_Inferring_Semantic_Layout_CVPR_2018_paper.pdf)] 
      * MC-GAN: Multi-conditional Generative Adversarial Network for Image Synthesis [[Paper](https://arxiv.org/pdf/1805.01123.pdf)] [[Code](https://github.com/HYOJINPARK/MC_GAN)]
      * Tell, Draw, and Repeat: Generating and Modifying Images Based on Continual Linguistic Instruction [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/El-Nouby_Tell_Draw_and_Repeat_Generating_and_Modifying_Images_Based_on_ICCV_2019_paper.pdf)] [[Code](https://github.com/Maluuba/GeNeVA)]
  * **StackGAN-based**
    * 2017: 
      * `StackGAN: Text to photo-realistic image synthesis with stacked generative adversarial networks` [[Paper](https://arxiv.org/pdf/1612.03242.pdf)] [[Code](https://github.com/hanzhanggit/StackGAN)]
    * 2018: 
      * StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks [[Paper](https://arxiv.org/pdf/1710.10916.pdf)] [[Code](https://github.com/hanzhanggit/StackGAN-v2)]
      * Text-to-image-to-text translation using cycle consistent adversarial networks [[Paper](https://arxiv.org/pdf/1808.04538.pdf)] [[Code](https://github.com/CSC2548/text2image2textGAN)]
      * AttnGAN: Fine-grained text to image generation with attentional generative adversarial networks [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf)] [[Code](https://github.com/taoxugit/AttnGAN)]
      * ChatPainter: Improving Text to Image Generation using Dialogue [[Paper](https://arxiv.org/pdf/1802.08216.pdf)] 
    * 2019:
      * 👸 FTGAN: A Fully-trained Generative Adversarial Networks for Text to Face Generation [[Paper](https://arxiv.org/abs/1904.05729)]
      * C4Synth: Cross-Caption Cycle-Consistent Text-to-Image Synthesis [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8658689)]
      * Semantics-Enhanced Adversarial Nets for Text-to-Image Synthesis [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tan_Semantics-Enhanced_Adversarial_Nets_for_Text-to-Image_Synthesis_ICCV_2019_paper.pdf)] 
      * Semantics Disentangling for Text-to-Image Generation [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yin_Semantics_Disentangling_for_Text-To-Image_Generation_CVPR_2019_paper.pdf)] [[Website](https://gjyin91.github.io/projects/sdgan.html)]
      * MirrorGAN: Learning Text-to-image Generation by Redescription [[Paper](https://arxiv.org/pdf/1903.05854.pdf)] [[Code](https://github.com/qiaott/MirrorGAN)]
      * Controllable Text-to-Image Generation [[Paper](https://papers.nips.cc/paper/2019/file/1d72310edc006dadf2190caad5802983-Paper.pdf)] [[Code](https://github.com/mrlibw/ControlGAN)]
      * DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis [[Paper](https://arxiv.org/pdf/1904.01310.pdf)] [[Code](https://github.com/MinfengZhu/DM-GAN)]
    * 2020:
      * CookGAN: Causality based Text-to-Image Synthesis [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_CookGAN_Causality_Based_Text-to-Image_Synthesis_CVPR_2020_paper.pdf)]
      * RiFeGAN: Rich Feature Generation for Text-to-Image Synthesis From Prior Knowledge [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_RiFeGAN_Rich_Feature_Generation_for_Text-to-Image_Synthesis_From_Prior_Knowledge_CVPR_2020_paper.pdf)] 
      * KT-GAN: Knowledge-Transfer Generative Adversarial Network for Text-to-Image Synthesis [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9210842)]
      * CPGAN: Content-Parsing Generative Adversarial Networks for Text-to-Image Synthesis [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58548-8_29)] [[Code](https://github.com/dongdongdong666/CPGAN)]
      * End-to-End Text-to-Image Synthesis with Spatial Constrains [[Paper](https://dl.acm.org/doi/pdf/10.1145/3391709)]
      * Semantic Object Accuracy for Generative Text-to-Image Synthesis [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9184960)] [[Code](https://github.com/tohinz/semantic-object-accuracy-for-generative-text-to-image-synthesis)]
    * 2021:
      * 👸 Multi-caption Text-to-Face Synthesis: Dataset and Algorithm [[Paper](https://dl.acm.org/doi/10.1145/3474085.3475391)] [[Code](https://github.com/cripac-sjx/SEA-T2F)]
      * 👸 Generative Adversarial Network for Text-to-Face Synthesis and Manipulation [[Paper](https://dl.acm.org/doi/abs/10.1145/3474085.3481026)]
      * 👸 Generative Adversarial Network for Text-to-Face Synthesis and Manipulation with Pretrained BERT Model [[Paper](https://ieeexplore.ieee.org/document/9666791)] 
      * Multi-Sentence Auxiliary Adversarial Networks for Fine-Grained Text-to-Image Synthesis [[Paper](https://ieeexplore.ieee.org/document/9345477)]
      * Unsupervised text-to-image synthesis [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320320303769)] 
      * RiFeGAN2: Rich Feature Generation for Text-to-Image Synthesis from Constrained Prior Knowledge [[Paper](https://ieeexplore.ieee.org/abstract/document/9656731)]
    * 2022:
      * 👸 DualG-GAN, a Dual-channel Generator based Generative Adversarial Network for text-to-face synthesis [[Paper](https://www.sciencedirect.com/science/article/pii/S0893608022003161)]
      * 👸 CMAFGAN: A Cross-Modal Attention Fusion based Generative Adversarial Network for attribute word-to-face synthesis [[Paper](https://www.sciencedirect.com/science/article/pii/S0950705122008863)]
      * DR-GAN: Distribution Regularization for Text-to-Image Generation [[Paper](https://arxiv.org/abs/2204.07945)] [[Code](https://github.com/Tan-H-C/DR-GAN-Distribution-Regularization-for-Text-to-Image-Generation)]
      * T-Person-GAN: Text-to-Person Image Generation with Identity-Consistency and Manifold Mix-Up [[Paper](https://arxiv.org/abs/2208.12752)] [[Code](https://github.com/linwu-github/Person-Image-Generation)]
  * **StlyeGAN-based**
    * 2021:
      * 👸 `TediGAN: Text-Guided Diverse Image Generation and Manipulation` [[Paper](https://arxiv.org/pdf/2012.03308.pdf)] [[Extended Version](https://arxiv.org/pdf/2104.08910.pdf)][[Code](https://github.com/IIGROUP/TediGAN)] [[Dataset](https://github.com/IIGROUP/Multi-Modal-CelebA-HQ-Dataset)] [[Colab](https://colab.research.google.com/github/weihaox/TediGAN/blob/main/playground.ipynb)] [[Video](https://www.youtube.com/watch?v=L8Na2f5viAM)] 
      * 👸 Faces a la Carte: Text-to-Face Generation via Attribute Disentanglement [[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Wang_Faces_a_la_Carte_Text-to-Face_Generation_via_Attribute_Disentanglement_WACV_2021_paper.pdf)]
      * Cycle-Consistent Inverse GAN for Text-to-Image Synthesis [[Paper](https://dl.acm.org/doi/10.1145/3474085.3475226)] 
    * 2022: 
      * 👸 Text-Free Learning of a Natural Language Interface for Pretrained Face Generators [[Paper](https://arxiv.org/abs/2209.03953)] [[Code](https://github.com/duxiaodan/Fast_text2StyleGAN)]
      * 👸 clip2latent: Text driven sampling of a pre-trained StyleGAN using denoising diffusion and CLIP [[Paper](https://arxiv.org/abs/2210.02347v1)] [[Code](https://github.com/justinpinkney/clip2latent)]
      * 👸 TextFace: Text-to-Style Mapping based Face Generation and Manipulation [[Paper](https://ieeexplore.ieee.org/abstract/document/9737433)]
      * 👸 AnyFace: Free-style Text-to-Face Synthesis and Manipulation [[Paper](https://arxiv.org/abs/2203.15334)] 
      * 👸 StyleT2F: Generating Human Faces from Textual Description Using StyleGAN2 [[Paper](https://arxiv.org/abs/2204.07924)] [[Code](https://github.com/DarkGeekMS/Retratista)]
      * 👸 StyleT2I: Toward Compositional and High-Fidelity Text-to-Image Synthesis [[Paper](https://arxiv.org/abs/2203.15799)] [[Code](https://github.com/zhihengli-UR/StyleT2I)]
      * LAFITE: Towards Language-Free Training for Text-to-Image Generation [[Paper](https://arxiv.org/abs/2111.13792)] [[Code](https://github.com/drboog/Lafite)] 
  * **Others**
    * 2018:
      * (Hierarchical adversarial network) Photographic Text-to-Image Synthesis with a Hierarchically-nested Adversarial Network [[Paper](https://arxiv.org/pdf/1802.09178.pdf)] [[Code](https://github.com/ypxie/HDGan)]
    * 2021:
      * (BigGAN) CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders [[Paper](https://arxiv.org/pdf/2106.14843.pdf)] [[Code](https://colab.research.google.com/github/kvfrans/clipdraw/blob/main/clipdraw.ipynb)]
      * (BigGAN) FuseDream: Training-Free Text-to-Image Generation with Improved CLIP+GAN Space Optimization [[Paper](https://arxiv.org/abs/2112.01573)] [[Code](https://github.com/gnobitab/FuseDream)]
    * 2022:
      * (One-stage framework) Text to Image Generation with Semantic-Spatial Aware GAN [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liao_Text_to_Image_Generation_With_Semantic-Spatial_Aware_GAN_CVPR_2022_paper.pdf)] [[Code](https://github.com/wtliao/text2image)]     
### <span id="head-transformer"> 3.2 **Autogressive** Model </span> [       «🎯Back To Top»       ](#)
  * **Transformer-based**
    * 2021:
      * `Zero-Shot Text-to-Image Generation` [[Paper](https://arxiv.org/pdf/2102.12092.pdf)] [[Code](https://github.com/openai/DALL-E)] [[Blog](https://openai.com/blog/dall-e/)] [[Model Card](https://github.com/openai/DALL-E/blob/master/model_card.md)] [[Colab](https://colab.research.google.com/github/openai/DALL-E/blob/master/notebooks/usage.ipynb)] [[Code(Pytorch)](https://github.com/lucidrains/DALLE-pytorch)]
      * CogView: Mastering Text-to-Image Generation via Transformers [[Paper](https://arxiv.org/pdf/2105.13290.pdf)] [[Code](https://github.com/THUDM/CogView)] [[Demo Website(Chinese)](https://lab.aminer.cn/cogview/index.html)] 
      * Unifying Multimodal Transformer for Bi-directional Image and Text Generation [[Paper](https://dl.acm.org/doi/10.1145/3474085.3481540)] [[Code](https://github.com/researchmm/generate-it)]
    * 2022: 
      * CogView2: Faster and Better Text-to-Image Generation via Hierarchical Transformers [[Paper](https://arxiv.org/abs/2204.14217)] [[Code](https://github.com/THUDM/CogView2)]
      * Scaling Autoregressive Models for Content-Rich Text-to-Image Generation [[Paper](https://arxiv.org/abs/2206.10789)] [[Code](https://github.com/google-research/parti)] [[Project](https://parti.research.google/)]
      * Neural Architecture Search with a Lightweight Transformer for Text-to-Image Synthesis [[Paper](https://ieeexplore.ieee.org/abstract/document/9699403)] 
      * DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers [[Paper](https://arxiv.org/abs/2202.04053)] [[Code](https://github.com/j-min/DallEval)] 
      * CLIP-GEN: Language-Free Training of a Text-to-Image Generator with CLIP [[Paper](https://arxiv.org/abs/2203.00386)] [[Code](https://github.com/HFAiLab/clip-gen)]
      * Text-to-Image Synthesis based on Object-Guided Joint-Decoding Transformer [[Paper](https://fengxianghe.github.io/paper/wu2022text.pdf)]
      * Autoregressive Image Generation using Residual Quantization [[Paper](https://arxiv.org/abs/2203.01941)] [[Code](https://github.com/kakaobrain/rq-vae-transformer)] 
      * Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors [[Paper](https://arxiv.org/abs/2203.13131)] [[Code](https://github.com/CasualGANPapers/Make-A-Scene)] [[The Little Red Boat Story](https://www.youtube.com/watch?v=N4BagnXzPXY)]
### <span id="head-diffusion"> 3.3 **Diffusion** Model </span> [       «🎯Back To Top»       ](#)
  * **Diffusion-based**
    * 2022:
      * `High-Resolution Image Synthesis with Latent Diffusion Models` [[Paper](https://arxiv.org/abs/2112.10752)] [[Code](https://github.com/CompVis/latent-diffusion)] [[Stable Diffusion Code](https://github.com/CompVis/stable-diffusion)]
      * Vector Quantized Diffusion Model for Text-to-Image Synthesis [[Paper](https://arxiv.org/abs/2111.14822)] [[Code](https://github.com/microsoft/vq-diffusion)]
      * Hierarchical Text-Conditional Image Generation with CLIP Latents [[Paper](https://cdn.openai.com/papers/dall-e-2.pdf)] [[Blog](https://openai.com/dall-e-2/)] [[Risks and Limitations](https://github.com/openai/dalle-2-preview/blob/main/system-card.md)] [[Unofficial Code](https://github.com/lucidrains/DALLE2-pytorch)] 
      * Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding [[Paper](https://arxiv.org/abs/2205.11487)] [[Blog](https://gweb-research-imagen.appspot.com/)]
      * GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models [[Paper](https://arxiv.org/abs/2112.10741)] [[Code](https://github.com/openai/glide-text2im)]
      * Compositional Visual Generation with Composable Diffusion Models [[Paper](https://arxiv.org/abs/2206.01714)] [[Code](https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch)] [[Project](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/)] [[Hugging Face](https://huggingface.co/spaces/Shuang59/Composable-Diffusion)]
      * Prompt-to-Prompt Image Editing with Cross Attention Control [[Paper](https://arxiv.org/abs/2208.01626)] [[Code](https://github.com/google/prompt-to-prompt)] [[Unofficial Code](https://github.com/bloc97/CrossAttentionControl)] [[Project](https://prompt-to-prompt.github.io/)]
      * Creative Painting with Latent Diffusion Models [[Paper](https://arxiv.org/abs/2209.14697)] 
      * DALL-E-Bot: Introducing Web-Scale Diffusion Models to Robotics [[Paper](https://arxiv.org/abs/2210.02438v1)] [[Project](https://www.robot-learning.uk/dall-e-bot)]
      * Swinv2-Imagen: Hierarchical Vision Transformer Diffusion Models for Text-to-Image Generation [[Paper](https://arxiv.org/abs/2210.09549)]
      * ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model with Knowledge-Enhanced Mixture-of-Denoising-Experts [[Paper](https://arxiv.org/abs/2210.15257)]
      * eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers [[Paper](https://arxiv.org/abs/2211.01324)] [[Project](https://deepimagination.cc/eDiffi/)] [[Video](https://www.youtube.com/watch?v=k6cOx9YjHJc)]
      * Multi-Concept Customization of Text-to-Image Diffusion [[Paper](https://arxiv.org/abs/2212.04488)] [[Project](https://www.cs.cmu.edu/~custom-diffusion/)] [[Code](https://github.com/adobe-research/custom-diffusion)] [[Hugging Face](https://huggingface.co/spaces/nupurkmr9/custom-diffusion)]
    * 2023:
      *  GLIGEN: Open-Set Grounded Text-to-Image Generation [[Paper](https://arxiv.org/abs/2301.07093)] [[Code](https://github.com/gligen/GLIGEN)] [[Project](https://gligen.github.io/)] [[Hugging Face Demo](https://huggingface.co/spaces/gligen/demo)] 
      *  Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis [[Paper (arXiv)](https://arxiv.org/abs/2212.05032)] [[Paper (OpenReview)](https://openreview.net/forum?id=PUIqjT4rzq7)] [[Code](https://github.com/shunk031/training-free-structured-diffusion-guidance)]
      *  Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models [[Paper](https://arxiv.org/abs/2301.13826)] [[Project](https://attendandexcite.github.io/Attend-and-Excite/)] [[Code](https://github.com/AttendAndExcite/Attend-and-Excite)] 
      *  Adding Conditional Control to Text-to-Image Diffusion Models [[Paper](https://arxiv.org/abs/2302.05543)] [[Code](https://github.com/lllyasviel/ControlNet)] 
      *  Editing Implicit Assumptions in Text-to-Image Diffusion Models[[Paper](https://arxiv.org/abs/2303.08084)] [[Project](https://time-diffusion.github.io/)] [[Code](https://github.com/bahjat-kawar/time-diffusion)] 

## <span id="head-4"> *4. Generative Applications* </span> [       «🎯Back To Top»       ](#)

### <span id="head-T2I"> 4.1 Text-to-Image </span> [       «🎯Back To Top»       ](#)

![Figure from paper](pic/Survey_CVPRW/Sec4_1_T2F.png)

Figure 1. **Diverse text-to-face results generated from GAN-based / Diffusion-based / Transformer-based models.** 
>Images in orange boxes are captured from original papers (a) [[zhou2021generative](https://ieeexplore.ieee.org/document/9666791)], (b) [[pinkney2022clip2latent](https://arxiv.org/abs/2210.02347v1)] and (c) [[li2022stylet2i](https://arxiv.org/abs/2203.15799)]; others are generated by a pre-trained model [[pinkney2022clip2latent](https://arxiv.org/abs/2210.02347v1)] [(b) left bottom row], [Dreamstudio](https://beta.dreamstudio.ai/generate) [(a-c) middle row] and [DALL-E 2](https://labs.openai.com/) [(a-c) right row] online platforms from textual descriptions. 

**Please refer to [Section 3 (Generative Models)](https://github.com/Yutong-Zhou-cv/Awesome-Text-to-Image/blob/main/%5BCVPRW%202023%F0%9F%8E%88%5D%20%20Best%20Collection.md#-3-generative-models---------back-to-top-------) for more details about text-to-image.**

### <span id="head-T2X"> 4.2 Text-to-X </span> [       «🎯Back To Top»       ](#)

![Figure from paper](pic/Survey_CVPRW/Sec4_2_T2X.png)

Figure 2. **Selected representative samples on Text-to-X.** 
>Images are captured from original papers ((a) [[ho2022imagen]()], (b)-Left [[xu2022dream3d]()], (b)-Right [[cite{poole2022dreamfusion]()], (c) [[tevet2022human]()]) and remade.

### <span id="head-X2I"> 4.3 X-to-Image </span> [       «🎯Back To Top»       ](#)

![Figure from paper](pic/Survey_CVPRW/Sec4_3_X2I.png)

Figure 3. **Selected representative samples on X-to-Image.** 
>Images are captured from original papers and remade. 

>(a) *Layered Editing* [[bar2022text2live]()] (Left), *Recontextualization* [[ruiz2023dreambooth]()] (Middle), *Image Editing* [[brooks2022instructpix2pix]()] (Right).   
>(b) *Context-Aware Generation* [[he2021context]()] (Left),  *Model Complex Scenes* [[yang2022modeling]()] (Right).   
>(c) *Face Reconstruction* [[dado2022hyperrealistic]()] (Left), *High-resolution Image Reconstruction* [[takagi2022high]()] (Right).   
>(d) *Speech to Image* [[wang2021generating]()] (Left), *Sound Guided Image Manipulation* [[lee2022robust]()] (Middle), *Robotic Painting* [[misra2023robot]()] (Right).   
>*Legend*: **X excluding “Additional Input Image”**  (Blue dotted line box, top row). **Additional Input Image**  (Green box, middle row). **Ground Truth**  (Red box, middle row). **Generated / Edited / Reconstructed Image**  (Black box, bottom row).

### <span id="head-multi"> 4.4 Multi Tasks </span> [       «🎯Back To Top»       ](#)


## <span id="head-5"> *5. Discussion* </span> [       «🎯Back To Top»       ](#)
