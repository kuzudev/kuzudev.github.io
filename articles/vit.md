# Vision Transformer - Basics

## Image to sequence

<p>Let’s recall that modern deep neural networks (DNNs) based on transformers operate on **sequences**.  
A sequence can represent:
- text,
- time series,
- or even patches of an image(s).
In our case, we will focus on **images**.</p> 
<br>
<p>Before being processed by a transformer-based DNN, an image typically undergoes the following steps:
1. **Splitting into patches**: the image is divided into fixed-size patches.
2. **Flattening patches**: each patch is converted into a 1D vector.
3. **Linear projection**: each vector is mapped into a higher-dimensional embedding space (forming a _token_).
4. Adding learnable positional embeddings, learnable CLS token (trained during training and fixed during inference).</p>

![1](vit/1.png)

$$
X = \left[ \; \vec w_{CLS} \;; \; \vec x_p^1 \; W_{LP} \; ; \; ... \; ; \; \vec x_p^N \; W_{LP} \; \right] + W_{positional \; embs}
$$

<p>$$\vec w_{CLS}$$ — learnable CLS token with shape $$1 \times D$$  
$$D$$ — dimension of embedding space  

$$H, W$$ — size of the input image  
$$P, P$$ — size of each patch  
$$N=\dfrac{HW}{P^2}$$ — number of patches  
$$\vec x_p^1, \ldots, \vec x_p^N$$ — flattened patches, each has shape \(1 \times P^2 C\)  

$$W_{LP}$$ — linear projection with shape $$P^2 C \times D$$  

$$W_{\text{positional}\; \text{embs}}$$ — learnable positional embeddings for each token, shape $$ (N+1) \times D $$</p>

## Attention

The resulting sequence is then fed into the transformer model.  
The essential building block of a transformer is the **attention**. See original paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762).

Attention allows transformers to maintain an extremely long-term memory: the model can _attend to_ all tokens in the sequence simultaneously.

In our case, this means that through attention, the transformer can “see” all parts of the image at once (sees with the resolution of the patch size). 
*This is the key difference from CNN-based models, where the receptive field in the 
first layers limits the view to local regions.*

### Idea

The attention mechanism can be described in terms of a _soft_ key–value database lookup.

Imagine we have a key-value database (for example, a simple Python dictionary) and we issue a query that contains a spelling mistake.

We want to compare the query $q$ with every key $k$ in the database, and return a weighted sum of the values.
The weight assigned to each value should correspond to the probability that the user actually asked the associated key.

By computing the vector of probabilities as
$$
softmax \left( similarity(q, k) \right)
$$
and then taking the dot product of this probability vector with the vector of values $v$, we obtain a "corrected" value from the database that best matches the asked meaning of the query $q$.

This explaination was taken [from this article](https://habr.com/ru/articles/599703/).


Let's we have our sequence obtained from the input image:
$$
X \in \Bbb R^{N+1 \; \times \; D}
$$
$$N$$ - number of patches
$$D$$ - length of each input embedding (each input token)

![1](vit/2.png)