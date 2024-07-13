# Sentiment Analysis about ChatGPT related to Academics Fields Using CNN-LSTM
![home](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/home-testing.png)

## Table of Contents
+ [Abstract](#abstract)
+ [Tech Stack](#techstack)
+ [Reports](#reports)
+ [Conclusion](#conclusion)
+ [Screenshots](#screenshots)
+ [References](#references)
+ [Author](#author)

## Abstract <a name = "abstract"></a>

ChatGPT is an intelligent chatbot capable of generate human-like text. The presence of ChatGPT in academic has sparked both pros and cons, with some considering it beneficial and others considering it as a threat. Many individuals, including academics, express their opinions through social media. These opinions can be used for evaluation and decision-making by academics in responding to ChatGPT in academic. The research objective is to create a sentiment analysis system to understand Indonesian public opinions on ChatGPT in the academic field through social media. The dataset used consists of 800 data consisting of three classes namely positive, negative, and neutral, each class consisting of 392, 243, and 165 respectively. The dataset is collected from social media platforms such as YouTube, Instagram, and Twitter. The model used to classify opinions is a model with a CNN-LSTM architecture, where the LSTM layer employed is a bidirectional LSTM. The best evaluation result was achieved with an accuracy of 79.17%. The precision, recall, and f1-score values of the model are 77.68%, 74.91%, and 75.86%. To compare models with different architectures, comparative testing was conducted against the LSTM-CNN, LSTM, and CNN architectures. The results show an accuracy of 77.49%, 75.47%, and 73.03%. Based on validation results with several academics, the sentiment analysis system has helped them in understanding public perceptions of ChatGPT in academic.

## Tech Stack <a name = "techstack"></a>

- Web Service (API): Flask
- Scraping: Selenium, Google API Client
- Modeling: Numpy, Pandas, Scikit-learn, Gensim, Tensorflow, Sastrawi, NLTK, googletrans, deep-translator, IndoNLP
- Web Application: Laravel 9, JQuery, Tailwind CSS, Flowbite, Vite, ApexCharts

## Reports <a name = "reports"></a>

Below is a table showing the evaluation metrics from the experiments conducted:

### Hyperparameter Searching Results on Data Preprocessing Experiments
<table>
    <tr>
        <th>Description</th>
        <th>Hyperparameter</th>
        <th>Performa</th>
        <th>Acc (%)</th>
        <th>Prec (%)</th>
        <th>Rec (%)</th>
        <th>F1 (%)</th>
    </tr>
    <tr>
        <td rowspan="2">Raw (without modification)</td>
        <td rowspan="2">
            Units: 96 <br>
            Dropout: 0.4 <br>
            Regularizer: L1
        </td>
        <td>Top Fold(3)</td>
        <td>60.15</td>
        <td>51.16</td>
        <td>51.93</td>
        <td>49.04</td>
    </tr>
    <tr>
        <td>Average</td>
        <td>58.5</td>
        <td>53.05</td>
        <td>47.8</td>
        <td>44.04</td>
    </tr>
    <tr>
        <td rowspan="2">Add a slang word dictionary</td>
        <td rowspan="2">
            Units: 64 <br>
            Dropout: 0.4 <br>
            Regularizer: L1
        </td>
        <td>Top Fold (1)</td>
        <td>61.05</td>
        <td>56.57</td>
        <td>50.54</td>
        <td>47.12</td>
    </tr>
    <tr>
        <td>Average</td>
        <td>59.5</td>
        <td>49.03</td>
        <td>48.48</td>
        <td>45.48</td>
    </tr>
        <tr>
        <td rowspan="2">Stopword dictionary modification</td>
        <td rowspan="2">
            Units: 64 <br>
            Dropout: 0.5 <br>
            Regularizer: L2
        </td>
        <td>Top Fold (3)</td>
        <td>63.53</td>
        <td>57.31</td>
        <td>54.7</td>
        <td>53.51</td>
    </tr>
    <tr>
        <td>Average</td>
        <td>60.75</td>
        <td>54.02</td>
        <td>51.33</td>
        <td>48.69</td>
    </tr>
    <tr>
        <td rowspan="2">Add data variety with data augmentation</td>
        <td rowspan="2">
            Units: 96 <br>
            Dropout: 0.5 <br>
            Regularizer: L2
        </td>
        <td>Top Fold (3)</td>
        <td><b>78.42</b></td>
        <td>77.38</td>
        <td>73.91</td>
        <td>74.98</td>
    </tr>
    <tr>
        <td>Average</td>
        <td>76.25</td>
        <td>74.41</td>
        <td>71.72</td>
        <td>72.64</td>
    </tr>
</table>

### Hyperparameter Searching Results on vectorization Model
<table>
    <tr>
        <th>Hyperparameter</th>
        <th>Performa</th>
        <th>Acc (%)</th>
        <th>Prec (%)</th>
        <th>Rec (%)</th>
        <th>F1 (%)</th>
    </tr>
    <tr>
        <td rowspan="2">
            Method: CBOW <br>
            Window Size: 5
        </td>
        <td>Top Fold (1)</td>
        <td>76.4</td>
        <td>75.62</td>
        <td>72.17</td>
        <td>73.4</td>
    </tr>
    <tr>
        <td>Average</td>
        <td>75.44</td>
        <td>74.66</td>
        <td>70.35</td>
        <td>71.74</td>
    </tr>
    <tr>
        <td rowspan="2">
            Method: CBOW <br>
            Window Size: 10
        </td>
        <td>Top Fold (3)</td>
        <td>77.86</td>
        <td>76.85</td>
        <td>73.8</td>
        <td>74.48</td>
    </tr>
    <tr>
        <td>Average</td>
        <td>76.0</td>
        <td>74.34</td>
        <td>71.66</td>
        <td>72.33</td>
    </tr>
     <tr>
        <td rowspan="2">
            Method: CBOW <br>
            Window Size: 15
        </td>
        <td>Top Fold (3)</td>
        <td>76.55</td>
        <td>74.28</td>
        <td>72.23</td>
        <td>72.94</td>
    </tr>
    <tr>
        <td>Average</td>
        <td>75.44</td>
        <td>73.18</td>
        <td>70.92</td>
        <td>71.62</td>
    </tr>
         <tr>
        <td rowspan="2">
            Method: Skip-gram <br>
            Window Size: 5
        </td>
        <td>Top Fold (3)</td>
        <td>78.42</td>
        <td>77.38</td>
        <td>73.91</td>
        <td>74.98</td>
    </tr>
    <tr>
        <td>Average</td>
        <td>76.25</td>
        <td>74.41</td>
        <td>71.72</td>
        <td>72.64</td>
    </tr>
        <td rowspan="2">
            Method: Skip-gram <br>
            Window Size: 10
        </td>
        <td>Top Fold (3)</td>
        <td>78.99</td>
        <td>79.4</td>
        <td>74.73</td>
        <td>75.86</td>
    </tr>
    <tr>
        <td>Average</td>
        <td>76.56</td>
        <td>75.25</td>
        <td>72.49</td>
        <td>73.31</td>
    </tr>
        </tr>
        <td rowspan="2">
            Method: Skip-gram <br>
            Window Size: 15
        </td>
        <td>Top Fold (3)</td>
        <td><b>79.17</b></td>
        <td>77.68</td>
        <td>74.91</td>
        <td>75.86</td>
    </tr>
    <tr>
        <td>Average</td>
        <td>76.63</td>
        <td>75.2</td>
        <td>72.09</td>
        <td>73.11</td>
    </tr>
</table>

### Comparison Results on Each Deep Learning Architecture

<table>
  <tr>
    <th>Architecture</th>
    <th>Performa</th>
    <th>Acc (%)</th>
    <th>Prec (%)</th>
    <th>Rec (%)</th>
    <th>F1 (%)</th>
  </tr>
  <tr>
    <td rowspan="2">LSTM</td>
    <td>Top Fold (1)</td>
    <td>49.25</td>
    <td>49.72</td>
    <td>33.54</td>
    <td>22.38</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>49.12</td>
    <td>38.58</td>
    <td>33.47</td>
    <td>22.21</td>
  </tr>
    <tr>
    <td rowspan="2">CNN</td>
    <td>Top Fold (1)</td>
    <td>73.03</td>
    <td>71.17</td>
    <td>68.34</td>
    <td>69.33</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>72.25</td>
    <td>69.71</td>
    <td>67.31</td>
    <td>68.09</td>
  </tr>
  <tr>
    <td rowspan="2">LSTM-CNN</td>
    <td>Top Fold (3)</td>
    <td>75.23</td>
    <td>72.96</td>
    <td>71.43</td>
    <td>71.62</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>74.56</td>
    <td>72.23</td>
    <td>70.59</td>
    <td>71.1</td>
  </tr>
  <tr>
    <td rowspan="2">CNN-LSTM</td>
    <td>Top Fold (3)</td>
    <td>67.73</td>
    <td>45.1</td>
    <td>57.11</td>
    <td>49.74</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>67.0</td>
    <td>51.24</td>
    <td>55.97</td>
    <td>49.39</td>
  </tr>
  <tr>
    <td rowspan="2">BiLSTM</td>
    <td>Top Fold (1)</td>
    <td>75.47</td>
    <td>73.44</td>
    <td>72.06</td>
    <td>72.66</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>75.06</td>
    <td>73.0</td>
    <td>71.37</td>
    <td>72.04</td>
  </tr>
  <tr>
    <td rowspan="2">BiLSTM-CNN</td>
    <td>Top Fold (3)</td>
    <td>77.49</td>
    <td>75.32</td>
    <td>75.69</td>
    <td>75.48</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>75.88</td>
    <td>73.51</td>
    <td>73.12</td>
    <td>73.23</td>
  </tr>
  <tr>
    <td rowspan="2">CNN-BiLSTM</td>
    <td>Top Fold (3)</td>
    <td>79.17</td>
    <td>77.68</td>
    <td>74.91</td>
    <td>75.86</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>76.63</td>
    <td>75.2</td>
    <td>72.09</td>
    <td>73.11</td>
  </tr>
</table>


## Conclusion <a name = "conclusion"></a>

From all the experiments conducted, the best model was obtained with performance metrics of accuracy, precision, recall, and F1-score of 79.17%, 77.68%, 74.91%, and 75.86%, respectively. The average accuracy, precision, recall, and F1-score of the model across all folds were 76.63%, 75.2%, 72.09%, and 73.11%, respectively. The comparison results also demonstrated that the CNN-LSTM architecture using bidirectional LSTM layers has better accuracy than other architectures. Based on user validation, it was found that the sentiment analysis system is functioning well and helps them understand the sentiment polarity of public opinions in Indonesia regarding ChatGPT in the academic field.

## Screenshots <a name = "screenshots"></a>

Here are some screenshots of the application:

![home-testing](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/home-testing.png)

![testing-result](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/testing-result.png)

![sentiment](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/sentiment.png)

![infographic](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/infographic.png)

![dashboard](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/dashboard.png)

![create-crawl](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/create-crawl.png)

![create-crawl-keywords](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/create-crawl-keywords.png)

![create-crawl-links](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/create-crawl-links.png)

![history-crawl](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/history-crawl.png)

![detail-crawl](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/detail-crawl.png)

![result-crawl](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/result-crawl.png)

![all-data](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/all-data.png)

![retrain-model](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/retrain-model.png)

![filter-modal](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/filter-modal.png)

![keywords](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/keywords.png)

![add-new-keywords](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/add-new-keywords.png)


![profiles](https://github.com/kevin-wijaya/resources/raw/main/images/sa-about-chatgpt-related-to-academics-fields-using-cnn-lstm/profiles.png)

## References <a name = "references"></a>

- Amriza, R. N. S., & Supriyadi, D. (2021). Komparasi Metode Machine Learning dan Deep Learning untuk Deteksi Emosi pada Text di Sosial Media. Jurnal JUPITER, 13(2), 130–139.
- Çerasi, C. Ç., & Balcioglu, Y. S. (2023, April). SENTIMENT ANALYSIS ON YOUTUBE: FOR CHATGPT. https://www.researchgate.net/publication/368850748_SENTIMENT_ANALYSIS_ON_YOUTUBE_FOR_CHATGPT
- ChatGPT “Maha Bisa”, Sebuah Solusi atau Masalah? (2023). https://telkomuniversity.ac.id/en/chatgpt-maha-bisa-sebuah-solusi-atau-masalah/
- Cui, Z., Ke, R., Pu, Z., & Wang, Y. (2018). Deep Bidirectional and Unidirectional LSTM Recurrent Neural Network for Network-wide Traffic Speed Prediction. 1–11. https://doi.org/10.48550/arXiv.1801.02143
- Firdaus, A. (2022). Aplikasi Algoritma K-Nearest Neighbor pada Analisis Sentimen Omicron Covid-19. Jurnal Riset Statistika, 2(2), 85–92. https://doi.org/10.29313/jrs.v2i2.1148
- Haque, M. U., Dharmadasa, I., Sworna, Z. T., Rajapakse, R. N., & Ahmad, H. (2022). “I think this is the most disruptive technology”: Exploring Sentiments of ChatGPT Early Adopters using Twitter Data. 1–12. https://doi.org/10.48550/arXiv.2212.05856
- Hermanto, D. T., Setyanto, A., & Luthfi, E. T. (2021). Algoritma LSTM-CNN untuk Binary Klasifikasi dengan Word2vec pada Media Online. Creative Information Technology Journal, 8(1), 64. https://doi.org/10.24076/citec.2021v8i1.264
- Kasneci, E., Sessler, K., Küchemann, S., Bannert, M., Dementieva, D., Fischer, F., Gasser, U., Groh, G., Günnemann, S., Hüllermeier, E., Krusche, S., Kutyniok, G., Michaeli, T., Nerdel, C., Pfeffer, J., Poquet, O., Sailer, M., Schmidt, A., Seidel, T., … Kasneci, G. (2023). ChatGPT for good? On opportunities and challenges of large language models for education. Learning and Individual Differences, 103, 1–13. https://doi.org/10.1016/j.lindif.2023.102274
- Kemp, S. (2023). DIGITAL 2023: INDONESIA. https://datareportal.com/reports/digital-2023-indonesia
- Munggaran, J. P., Alhafidz, A. A., Taqy, M., Aprianti, D., Agustini, R., & Munawir, M. (2023). Sentiment Analysis of Twitter Users’ Opinion Data Regarding the Use of ChatGPT in Education. 2(2), 75–88. https://doi.org/10.17509/coelite.v2i2.59645
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. Phylogenetic Diversity: Applications and Challenges in Biodiversity Science, 2(2), 1310–1318. https://doi.org/10.48550/arXiv.1211.5063
- Priyadarshini, I., & Cotton, C. (2021). A novel LSTM–CNN–grid search-based deep neural network for sentiment analysis. Journal of Supercomputing, 77(12), 13911–13932. https://doi.org/10.1007/s11227-021-03838-w
- Sharma, A. K., Chaurasia, S., & Srivastava, D. K. (2020). Sentimental Short Sentences Classification by Using CNN Deep Learning Model with Fine Tuned Word2Vec. Procedia Computer Science, 167(2019), 1139–1147. https://doi.org/10.1016/j.procs.2020.03.416
- Tanggapi Isu ChatGPT, FISIPOL UGM Gelar Sarasehan Bertajuk Polemik ChatGPT. (2023). https://fisipol.ugm.ac.id/tanggapi-isu-chatgpt-fisipol-ugm-gelar-sarasehan-bertajuk-polemik-chatgpt/
- Widhiyasana, Y., Semiawan, T., Mudzakir, G. I. A., & Noor, M. R. (2021). Penerapan Convolutional Long Short-Term Memory untuk Klasifikasi Teks Berita Bahasa Indonesia (Convolutional Long Short-Term Memory Implementation for Indonesian News Classification). Jurnal Nasional Teknik Elektro Dan Teknologi Informasi |, 10(4), 354–361. https://doi.org/10.22146/jnteti.v10i4.2438



## Author <a name = "author"></a>
- **Kevin Wijaya** 