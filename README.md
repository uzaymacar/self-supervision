# Self-Supervision

Self-supervised learning exploits learning signals that come free with the data as opposed to human 
annotations which are expensive to collect, not readily available, and often noisy. Self-supervised 
learning has revolutionized machine learning across many modalities and tasks, most notably helping 
transformers achieve state-of-the-art results in natural language understanding tasks.

Here, we provide implementations of several self-supervised pretext tasks for language and vision
modalities. For language, we have implemented i) *masked language modeling (MLM)* as in BERT [1], ii) *sequential
MLM*, iii) *permutation-based MLM* as in XLNet [2], iv) *text-infilling* as in SPANBERT [3] and BART [4], 
v) *token deletion* as in BART [4], vi) *document rotation* as BART [4], and v) *permutation* as in BART [4].
For vision, we have implemented i) *rotation* [5], ii) *counting* [6], iii) *context encoder* [7], and iv) 
*jigsaw puzzle* [8].

You can find examples respectively in `language/examples.ipynb` and `vision/examples.ipynb`.

Each method is realized through a data loader with a custom PyTorch data collator that returns a batch with
a set of inputs `x`, and a set of pseudo labels `y` in each iteration. You can find these respectively
in `language/collators.py` and `vision/collators.py`. Additionally, we provide model and loss function
implementations for vision methods based on the papers. You can find these in `vision/models.py` and
`vision/losses.py`.

## References

[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, & Kristina Toutanova. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 

[2] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, & Quoc V. Le. (2020). XLNet: Generalized Autoregressive Pretraining for Language Understanding. 

[3] Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, & Omer Levy. (2020). SpanBERT: Improving Pre-training by Representing and Predicting Spans. 

[4] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, & Luke Zettlemoyer. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. 

[5] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations, 2018.

[6] Mehdi Noroozi, Hamed Pirsiavash, and Paolo Favaro. Representation learning by learning to count, 2017.

[7] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A. Efros. Context encoders: Feature learning by inpainting, 2016.

[8] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles, 2017.
