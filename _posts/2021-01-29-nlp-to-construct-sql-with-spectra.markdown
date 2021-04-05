---
layout: post
title: Natural Language to SQL with Spectra
date: 2021-01-29
description: # 
img: ../graphics/spectra/cover.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [nlp, glove, vae] # add tag
usemathjax: true
---
    
In the [ITT13](https://people.bath.ac.uk/mtp34/itt13.html) event, [Spectra Analytics](http://www.spectraanalytics.com/) presented two challenges they have. The one that interested me was a natural language problem as follows. Imagine a client has a database with some number of tables and relationships, and they would like to query the database to extract some slice of the data they have. To query it, they have to write an SQL query which now requires an in-house SQL specialist. As the majority of people wanting to query this database are from a non-SQL background, the SQL specialist has to craft a variety of SQL queries where they will need to understand exactly what the data structure is and what the people actually want when they pose the natural language questions to them.

<p align="center">
  <img src="../assets/graphics/spectra/problem.png">
  <figcaption><p align="center"><strong>Figure 1. </strong>The problem Spectra's clients are facing.</p></figcaption>
</p>

Further they are also interested in ensuring two similar natural language questions should ouput two similar SQL queries (or even the same). During the ITT event, my group only really had two days to work on this problem and so we did not solve it but developed a pipeline with a few proof of concept ideas. To break down this task into modular components we need to:

1. understand individual words and project them into some vector space
2. understand individual questions and project them into some vector space
3. develop a model that can go from a question vector space to a SQL query text output

The first can be tackled by something called word [word embeddings](https://en.wikipedia.org/wiki/Word_embedding). There are a multitude of word embeddings out there. The simplest one would be to one-hot encode the words but then this would assume that words are independent to each other when in reality, words can be synonomous to other words and so there is no complete independence. Other embeddings usually involved a neural network but in the interest in showing some preliminary results our group would prefer something not too computationally expensive. For this reason, I suggested the use of [GloVe](https://www.aclweb.org/anthology/D14-1162/) (global vectors).


## GloVe

This technique looks at how often words appear next to other words termed **co-occurance** in the original paper, and projects each word into some vector space based on the co-occurance matrix. Words that are usually surrounded by similar words (or used in similar context) should be projected into similar areas of the vector space.

The optimisation involves minimising

\begin{equation}
    J = \sum_{i,j} f(x_{ij})(\mathbf{w}\_{i}^\text{T} \tilde{\mathbf{w}}\_{j} - \log x\_{ij})^2
\end{equation}

where $x_{ij}$ is the number of co-occurances of word $i$ occuring with word $j$, $\mathbf{w}_i$ is the $i$-th word vector, and $\tilde{\mathbf{w}}_j$ is the context word vector. The word and context word vectors should be equivalent and should only differ (by a small amount) due to random initialisation. More details can be found in the [original paper](https://www.aclweb.org/anthology/D14-1162/).

When executing this on a Wikipedia corpus, we found that the more dimensions in this word vector space, the better the nearest words to "science" becomes...until a certain point. Thereafter, it seems like we lose the natural clustering in vector space which is to be expected. If the number of dimensions in the vector space is the number of words, it is possible for each word to be prodominently projected into their own dimension which is something we do not want.

|   | 2         | 10          | 50         | 100        | 300        | 600           |
|---|-----------|-------------|------------|------------|------------|---------------|
| 1 | man       | studies     | study      | fiction    | fiction    | fiction       |
| 2 | created   | study       | scientific | study      | scientific | technology    |
| 3 | short     | historical  | society    | scientific | technology | scientific    |
| 4 | according | art         | fiction    | research   | study      | poppi         |
| 5 | half      | articles    | studies    | sciences   | research   | annuallymales |
| 6 | written   | scientific  | creation   | literature | sciences   | navadvipa     |
| 7 | t         | research    | research   | creation   | scientists | thacker       |
| 8 | along     | works       | literature | work       | creation   | osteopathic   |
| 9 | original  | mathematics | sciences   | philosophy | philosophy | norio         |


<p align="center"><strong>Table 1. </strong> The nine closest words to "science" in GloVe space as the number of embedding dimensions varies.</p>


From the above table, it seems like 600 dimensional word vector spaces produce weird results, and a dimension size of 2 also produces weird results. From visual inspection we think that 300 is a good size and reading around the literature that uses GloVe, they also seem to agree that 300 is a good size.

So now we have a way of clustering similar words in isolation in some vector space, we need a way of doing this with natural language questions. Ideally, we would also be able to assign probabilities to our outputs so we can sample SQL queries and have uncertainty quantification. From this angle we consider the use of Variational Autoencoders for this task.


# Variational Autoencoder


A VAE attempts to learn the topology of your data in some latent space. There are two objectives the VAE is optimised to satisfy. The output of the VAE should be roughly equal to the input, and the lower dimensional representation in the middle needs to look like a Gaussian. To state the objectives a bit more mathematically, first defining $p : \mathbf{x} \rightarrow \mathbf{z}$ to be the probabilistic projection into the lower dimensional space $\mathbf{z}$ from the original data input $\mathbf{x}$ with parameters to tune for the encoder function, and secondly $q : \mathbf{z} \rightarrow \mathbf{x}$ to be the probabilistic reconstruction from latent space to data space with parameters to tune for the decoder, then we wish to minimise

\begin{equation}
    \mathcal{L}\big(\mathbf{x}, q(p(\mathbf{x}))\big) + \text{D}_{\text{KL}}\big(p(\mathbf{z}\mid\mathbf{x})\mid\mid p(\mathbf{z})\big)
\end{equation}

where $\mathcal{L}$ is the time-distributed categorical cross-entropy, and $p(\mathbf{z})$ is a standard Gaussian.

<p align="center">
  <img src="../assets/graphics/spectra/vae.png" width="400 %">
  <figcaption><p align="center"><strong>Figure 2. </strong>VAE pipeline. We use the GloVe embeddings learnt to attempt to compress natural language questions into some lower dimensional space which we can sample from.</p></figcaption>
</p>

Implementing in TensorFlow (Keras), we have the following summary.

<p align="center">
  <img src="../assets/graphics/spectra/vae-summary.png" width="600 %">
  <figcaption><p align="center"><strong>Figure 3. </strong>VAE architecture summary.</p></figcaption>
</p>

Note that this is a fairly small network relative to the models published in the context of NLP but for the purposes of obtaining quick preliminary results within the event, we use this small network and only train for 100 iterations. From the above, we see that we compress the natural language questions into 32 dimensional space. This again was just to speed up computations but perhaps it would be better for the dimensionality of a natural languge input to be higher than the dimensionality of each word i.e. this compressed space dimensionality should exceed 300 (dimensionality of each word). Note that the network was trained using Quora questions from the [Quora Questions Kaggle competition](https://www.kaggle.com/c/quora-question-pairs) so all natural language embeddings are in the context of questions.

After training the model for 100 iterations, we randomly sample in the 32 dimensional natural language latent space (according to a standard Gaussian) and we yield the following numpy array.

<p align="center">
  <img src="../assets/graphics/spectra/sample-latent.png" width="600 %">
  <figcaption><p align="center"><strong>Figure 4. </strong>Four sampled points in 32 dimensional natural language latent space.</p></figcaption>
</p>

Using the decoder part of our VAE, we yield the following questions.

<p align="center">
  <img src="../assets/graphics/spectra/sampled-sentences.png" width="600 %">
  <figcaption><p align="center"><strong>Figure 5. </strong>Reconstructed sentences from random samples in latent space.</p></figcaption>
</p>

More interestingly, we can interpolate between these sentences (in latent space) to check if the nearby points are somewhat related as well as how to interpolate between questions.

<p align="center">
  <img src="../assets/graphics/spectra/interpolated-sentences.png" width="600 %">
  <figcaption><p align="center"><strong>Figure 6. </strong>Reconstructed interpolated sentences from random samples in latent space.</p></figcaption>
</p>

We can observe that some questions do not quite make sense but that is to be expected with a small network that only trained for 100 iterations. This is just a proof of concept that my team wanted to show Spectra.

## Summary

When presenting this work to Spectra, they seem pleased with what was accomplished over two days. The full pipeline proposed includes a ranked output (based on probability within the latent natural language space) to give a continuous feedback for the new decoder to learn.

<p align="center">
  <img src="../assets/graphics/spectra/full-pipeline.png" width="1500 %">
  <figcaption><p align="center"><strong>Figure 7. </strong>Full pipeline. Yellow boxes represent datasets required.</p></figcaption>
</p>

The full training procedure can be summarised as follows:

+ Train a word embedding model (GloVe was used here but there are plenty others to consider) to cluster words of similar context together
+ Train a VAE model (ideally a much larger network until convergence) to learn a natural language embedder
+ Train a separate decoder to target SQL queries from natural language questions (example dataset could be [spider 1.0](https://yale-lily.github.io/spider))
+ Once pre-trained on these datasets, give it to clients to continually update the model with their personalised use cases

## References

+ J. Pennington, R. Socher, C. D. Manning, *GloVe: Gloval Vectors for Word Representation*, 2014, [link](https://www.aclweb.org/anthology/D14-1162.pdf)
+ D. P. Kingma, M. Welling, *Auto-Encoding Variational Bayes*, 2014, [link](https://arxiv.org/abs/1312.6114)

## Further Reading

+ I. Bhalla, A. Gupta, *Generating SQL Queries from Natural Language*, 2018, [link](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6907018.pdf)
+ M. Peters, M. Neumann, M. Iyyer, M, Gardner, C. Clark, K. Lee,L. Zettlemoyer, *Deep Contextualized Word Representations*, 2018, [link](https://www.aclweb.org/anthology/N18-1202/)
+ I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, .B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio, *Generative Adversarial Networks*, 2014, [link](https://arxiv.org/abs/1406.2661)
+ J. Zhu, T. Park, P. Isola, A. A. Efros, *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, 2017, [link](https://arxiv.org/abs/1703.10593)




