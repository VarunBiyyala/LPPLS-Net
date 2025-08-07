# LPPLS-Net: Fast and Accurate Forecasting of Critical Points in Financial Time Series

This repository contains the official poster presented at the **SIAM Conference on Applications of Dynamical Systems 2025**, showcasing our research on **LPPLS-Net**, a neural network model for forecasting critical points (crashes) in noisy financial markets.

## 📌 Poster

📄 [Download Poster (PDF)](./LPPLS_Net__SIAM_Poster.pdf)

The poster provides a concise summary of our proposed method, synthetic data generation, topological validation, and evaluation on real-world market scenarios (Dot-com and Lehman collapse).

## 📚 Abstract

Forecasting financial crashes is vital but challenging.  
The Log-Periodic Power Law Singularity (LPPLS) model captures bubble dynamics but suffers from unstable parameter fitting.

We propose **LPPLS-Net**, a deep neural architecture trained on synthetically generated noisy time series (using white, AR(1), and GARCH(1,1) noise).  
Our method achieves fast and accurate detection of critical points, with performance validated using Topological Data Analysis (TDA) and evaluated on real-world scenarios from NASDAQ-100 and S&P 500.

## 🧠 Key Contributions

- Neural forecasting model trained on synthetic LPPLS + noise trajectories.
- Use of **TDA-based filtering** to ensure topological consistency of training data.
- **Curriculum learning** and **soft-penalty losses** for stable and accurate parameter recovery.
- Validated on Dot-com and Lehman crash periods using real market data.

## 🚀 Getting Started: Demo Notebook

To quickly try LPPLS-Net, see [`demo.ipynb`](./demo.ipynb).  
This notebook provides:

- Setup and installation instructions
- Example code to generate or load data
- Step-by-step guide to train and using the LPPLS-Net model
- Visualization and experimentation tips

**How to use:**
- Run `demo.ipynb` locally or in [Google Colab](https://colab.research.google.com/) for an interactive experience.

## 🔗 Links

- 📜 [About TDA](https://arxiv.org/pdf/1703.04385))
- 📈 [Learn more about LPPLS](https://arxiv.org/pdf/cond-mat/9901035)

## 🤝 Connect With Us

[![LinkedIn](https://img.shields.io/badge/Varun%20Biyyala-LinkedIn-blue)](https://www.linkedin.com/in/varunbiyyala/)  
[![LinkedIn](https://img.shields.io/badge/Dr.%20Marian%20Gidea-LinkedIn-blue)](https://www.linkedin.com/in/marian-gidea-42822a29/)

## 📬 Contact

If you have any questions or would like to collaborate, feel free to reach out via LinkedIn.

---

© 2025 Varun Biyyala & Marian Gidea  
This work was presented at the **SIAM Conference on Applications of Dynamical Systems**, Miami, 2025.
