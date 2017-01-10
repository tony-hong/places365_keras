# Places365 on VGG16

A VGG16 model, trained on Places365 model in Keras, with support for finetuning on this model.


### Running it

* Download the main model by running `bash download_model.sh`
* As a standalone file, run `python place.py <image>`
* To use as a module, call predict(image_path, return_dict={}), to receive the results in `return_dict`

### Finetuning on this model

* To finetune this model, run `python finetune_vgg16.py <images_directory>`, where <images_directory> is a path to a directory with images, with subfolders with names of the classes, and images inside them.

### Reference
Link: [Places2 Database](http://places2.csail.mit.edu), [Places1 Database](http://places.csail.mit.edu)

Please cite the following paper(https://arxiv.org/pdf/1610.02055.pdf) if you use the data or pre-trained CNN models.

```
@article{zhou2016places,
  title={Places: An Image Database for Deep Scene Understanding},
  author={Zhou, Bolei and Khosla, Aditya and Lapedriza, Agata and Torralba, Antonio and Oliva, Aude},
  journal={arXiv preprint arXiv:1610.02055},
  year={2016}
}
```

### Acknowledgements and License

Places dataset development has been partly supported by the National Science Foundation CISE directorate (#1016862), the McGovern Institute Neurotechnology Program (MINT), ONR MURI N000141010933, MIT Big Data Initiative at CSAIL, and Google, Xerox, Amazon and NVIDIA. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation and other funding agencies. 

The pretrained places-CNN models can be used under the Creative Common License (Attribution CC BY). Please give appropriate credit, such as providing a link to our paper or to the [Places Project Page](http://places2.csail.mit.edu). The copyright of all the images belongs to the image owners.
