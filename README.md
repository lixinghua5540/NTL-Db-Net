# NTL-Db-Net

Title: ***Making the Earth Clear at Night: A High-resolution Nighttime Light Image Deblooming Network***

<br>
<br>

***Abstract***<br>
High-resolution nighttime light (HR-NTL) imagery is a reliable indicator of human activity in urban areas at night. However, HR-NTL images on Earth are blurred because of issues such as blooming, noise, and overexposure in the imaging system and transmission path. Toward this end, a nighttime light image deblooming network (NTL-Db-Net) is proposed, which aims to make the Earth clearer at night. The NTL-Db-Net employs a generative adversarial network (GAN) as its deblooming framework. As the lack of ideal clear NTL data for GAN training, our method integrates blur transfer algorithm to simulate degradation for generating training data. This method was tested using HR-NTL data on Wuhan, Macau, Changchun, Xiamen, Atlanta and Jerusalem from HR Jilin-1 satellite constellation. Compared with the results of state-of-the-art SEAM, Uformer, and DRBNet, those from NTL-Db-Net demonstrated superior visual features and achieved the highest scores across all selected non-reference evaluation metrics. NTL-Db-Net also exhibited a lower number of network parameters and a smaller time cost. In addition, this study conducted a spatial clustering analysis of nighttime light with road networks and permeable surfaces. The results indicated that nightlights and roads exhibited a distinct spatially aggregated distribution, while permeable surfaces displayed a clear spatially dispersed distribution.
<br>
![image](https://github.com/lixinghua5540/NTL-Db-Net/blob/master/images/%E5%9B%BE2%E6%B5%81%E7%A8%8B%E5%9B%BE%E6%80%BB%E8%A7%88.jpg)
<br>
<br>
![image](https://github.com/lixinghua5540/NTL-Db-Net/blob/master/images/%E5%9B%BE2%E6%B5%81%E7%A8%8B%E5%9B%BE%E6%80%BB%E8%A7%88.jpg)
<br>Fig.2 methodological process
<br>
<br>
***Data Preparation***<br>
The data used in the experiments were purchased and are not disclosed here.<br>
Users can purchase the HR-NTL image data, and place them in the folders under . /dataset.
<br>
<br>
***Usage***
<br>
pytorch<br>
run train.py
