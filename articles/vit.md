Let’s recall that modern deep neural networks (DNNs) based on transformers operate on **sequences**.  
A sequence can represent:
- text,
- time series,
- or even patches of an image(s).
In our case, we will focus on **images**. 

Before being processed by a transformer-based DNN, an image typically undergoes the following steps:
1. **Splitting into patches**: the image is divided into fixed-size patches.
2. **Flattening patches**: each patch is converted into a 1D vector.
3. **Linear projection**: each vector is mapped into a higher-dimensional embedding space (forming a _token_).
4. Adding learnable positional embeddings, learnable CLS token (trained during training and fixed during inference).

![1](vit/1.png)

$$
X = \left[ \; \vec w_{CLS} \;; \; \vec x_p^1 \; W_{LP} \; ; \; ... \; ; \; \vec x_p^N \; W_{LP} \; \right] + W_{positional \; embs}
$$

$$\vec w_{CLS}$$ — learnable CLS token with shape $$1 \times D$$  
$$D$$ — dimension of embedding space  

$$H, W$$ — size of the input image  
$$P, P$$ — size of each patch  
$$N=\dfrac{HW}{P^2}$$ — number of patches  
$$\vec x_p^1, \ldots, \vec x_p^N$$ — flattened patches, each has shape \(1 \times P^2 C\)  

$$W_{LP}$$ — linear projection with shape $$P^2 C \times D$$  

$$W_{\text{positional}\; \text{embs}}$$ — learnable positional embeddings for each token, shape $$ (N+1) \times D $$