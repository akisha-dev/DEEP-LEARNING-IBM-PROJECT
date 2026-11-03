# IMAGE GENERATION USING DEEP CONVOLUTIONAL GAN (DCGAN)
**Synthesising Realistic MNIST Digits and Diagnosing Training Instability**

## Project Overview
Generating realistic synthetic images is a central problem in modern machine learning. This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) from scratch in PyTorch to synthesise handwritten digits from the MNIST dataset. The primary objective is to track adversarial loss dynamics across epochs and diagnose training instability utilizing stabilisation techniques such as one-sided label smoothing and tuned learning rates.

## Team Members (Group - 5)
* **Ankit Mishra** (FET-BAML-2024-28-017)
* **Subhadip Mani** (FET-BAML-2024-28-018)
* **Dibya Prakash Das** (FET-BAML-2024-28-019)
* **Akanksha Priya** (FET-BAML-2024-28-021)
* **Guide:** Yugansh Varshney 

## Tech Stack & Environment
* **Language:** Python 3
* **Frameworks:** PyTorch 2.x, Torchvision
* **Libraries:** NumPy, Matplotlib
* **Environment:** Google Colab (NVIDIA Tesla T4 GPU)

## Dataset
* **Dataset:** MNIST Training Set (60,000 grayscale images)
* **Resolution:** 28x28 pixels
* **Preprocessing:** Tensors scaled to a `[-1, 1]` range via `transforms.Normalize((0.5,), (0.5,))` to match the generator's Tanh output layer.

## Model Architecture
The architecture strictly follows DCGAN guidelines, utilizing strided/transposed convolutions and omitting fully connected hidden layers to prevent mode collapse.

**Generator (G)**
* **Input:** 100-dimensional Latent Noise Vector
* **Block 1:** ConvTranspose2d (128x7x7) + BatchNorm + ReLU
* **Block 2:** ConvTranspose2d (64x14x14) + BatchNorm + ReLU
* **Output Layer:** ConvTranspose2d + Tanh (1x28x28 image)

**Discriminator (D)**
* **Input:** 1x28x28 Image
* **Block 1:** Conv2d (64x14x14) + LeakyReLU
* **Block 2:** Conv2d (128x7x7) + BatchNorm + LeakyReLU
* **Output Layer:** Flatten + Linear (1 Logit output)

## Training Configuration
* **Batch Size:** 128
* **Optimizer:** Adam (β1 = 0.5, β2 = 0.999)
* **Loss Function:** Binary Cross Entropy with Logits (`BCEWithLogitsLoss`)
* **Epochs:** 20
* **Stabilisation Techniques Tested:** One-sided label smoothing (0.9 target for real images), dynamic Learning Rates (0.0001, 0.0002, 0.0003).

## Key Results & Observations
The DCGAN successfully generated highly recognisable handwritten digits by Epoch 5 and maintained structural fidelity through Epoch 20. 

* **Mode Collapse:** A pixel-diversity score of ~0.38 was maintained across all configurations, mathematically confirming a complete absence of mode collapse.
* **Label Smoothing:** Applying a 0.9 target for real images successfully raised the discriminator's loss floor (stabilizing at 0.7764 vs. baseline 0.6284), preventing the discriminator from becoming overconfident and overpowering the generator.
* **Learning Rate Dynamics:** Tuning the learning rate to 0.0001 provided smoother transitions between epochs, while 0.0003 resulted in sharper, noisier adversarial loss spikes.

## How to Run
1. Clone the repository and open `DCGAN_Project.ipynb` in Google Colab or Jupyter Notebook.
2. Ensure you have an active GPU runtime (e.g., Tesla T4).
3. Run the **Setup** cell to import PyTorch and define hyperparameters.
4. Execute the **Data Loading**, **Generator**, and **Discriminator** cells to initialize the architecture.
5. Run the **Training Loop** to begin the adversarial minimax game. Visual sample grids will automatically generate and save at epochs 1, 5, 10, 15, and 20.
6. Use the **Diagnostics & Results** section to plot Discriminator (L_D) and Generator (L_G) loss curves.
