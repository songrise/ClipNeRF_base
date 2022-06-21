# About
This repo contains my work as a Research Assistant (Intern) at CityU, Hong Kong.


In this repo I am trying to develop a very naive baseline method that utilize [CLIP](https://openai.com/blog/clip/) to manipulate the radiance field predicted by NeRF. For a related literature, please view [Clip-NeRF](https://scholar.google.com/scholar_url?url=https://openaccess.thecvf.com/content/CVPR2022/html/Wang_CLIP-NeRF_Text-and-Image_Driven_Manipulation_of_Neural_Radiance_Fields_CVPR_2022_paper.html&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&d=9170708679138992367&ei=e0ioYr_6LP6J6rQPzPSs-A8&scisig=AAGBfm2SrupHdCRswwklFZlswIm2qOlCow). Note that the above paper is not the exact one I am reproducing in this repo, there are some technical difference. To be more specific, we are trying to perform cross-domain scene-editing using a "specific" method (~~not gonna tell you which & how~~).

Overall, the code in this project is not programmed with maintainability in mind. It is just to validate an idea. Feel free to contact me if you have any questions or discussion.

![source](img/room.gif)

![source](img/room_cubism.gif)

# Installation guide
The baseline method are build upon two nerf implementations: [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) and [nerf-pl](https://github.com/kwea123/nerf_pl). I have implemented the same baseline method with these two nerf code, and both has been tested on llff dataset. However, I will not maintain the nerf-pytorch implementation anymore.

If you would like to play with the code, follow the enviroment setup guide in the readme of either nerf implementation. Also remember to check my `requirement.txt`. Then, you can refer to the argument in `run_nerf_dir.py`(for nerf-pytorch) or `opts.py`(for nerf_pl)  and train your stylized scene.
