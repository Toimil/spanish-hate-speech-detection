
## 🌟 Overview

This repository presents a comprehensive study on **automatic hate speech detection in Spanish social media**, with a particular focus on **misogyny** and **anti-immigrant** hate.

The work combines **linguistic and psycholinguistic analysis** with extensive experimentation across **classical machine learning models**, **Transformer-based architectures**, and **Large Language Models** (LLMs). In addition, we explore the impact of preprocessing strategies, hyperparameter optimization, and domain adaptation techniques based on guided masking with hate-related lexicons.

Experiments are conducted on three benchmark datasets:

- **AMI (IberEval 2018)**

- **HatEval (SemEval 2019)**

- **MisoCorpus 2020**

The proposed approaches achieve results that outperform previous competition winners and subsequent work, highlighting the importance of robust preprocessing pipelines, adaptation mechanisms, and careful model selection and hyperparameter optimization


---

## 📂 Repository Structure

The repository is organized into the following main components:

### 1. 🔬 `analysis/`
Linguistic, semantic, and psycholinguistic analyses of hate speech datasets.

This directory includes notebooks for:

- **Emotion analysis**
- **Sentiment analysis**
- **Irony detection**
- **Part-of-Speech (POS) analysis**
- **Named Entity Recognition (NER)**
- **Lexical statistics**, including word frequency, hashtags, and emojis

📌 Goal: Complement classification results with deeper linguistic insights into misogynistic and anti-immigrant hate speech.

### 2. 📊 `data/`  
This directory is intentionally left empty due to dataset licensing and ownership restrictions. Access to the data must be requested from the original dataset providers.

### 3. 🧹 `preprocess/`  
Text preprocessing pipelines.

📌 Goal: Analyze the impact of preprocessing decisions on downstream performance.

### 4. 🤖 `models/`  
Implementation of all modeling strategies evaluated in this work.

🔹 **Traditional Models**

Located in `models/traditional_models/`:
- Support Vector Machines (linear and RBF kernels)
- XGBoost

📌 Goal: Establish strong classical baselines.

🔹 **Hyperparameter Optimization**

Located in `models/hyperparameter_optimization/`:

Systematic hyperparameter search for Transformer-based models.

📌 Goal: Maximize classification performance by identifying optimal hyperparameter configurations for models.

🔹 **Best Model Selection**

Located in `models/test_with_best_val_models/`:

Evaluation using the best-performing configurations on validation data.

🔹 **Domain Adaptation**

Located in `models/domain_adaptation/`:

- Guided masking techniques using hate-related lexicons
- Adaptation of pretrained models (BETO, RoBERTuito)

📌 Goal: Improve model robustness by adapting general-purpose language models to hate speech domains.

🔹 **Zero-shot and Few-shot Learning**

Located in `models/zero_few_shot/`:

- Zero-shot and few-shot experiments using GPT-4o model

📌 Goal: Assess the ability of LLMs to perform hate speech detection without task-specific fine-tuning.


### 5. 🧪 `error_analysis/`  
Error analysis of model predictions.

- Inspection of misclassified instances
- Analysis of false positives and false negatives

📌 Goal: Understand model limitations and sources of error in hate speech detection.

---

## 📑 Citation

Coming soon!

---

## ⚠️ Disclaimer 

This work contains examples of offensive, hateful, or otherwise harmful language used in real-world social media content. These examples are included strictly for research and analytical purposes. The views or sentiments expressed in the content do not reflect those of the authors.
Reader discretion is advised, as some content may be disturbing or offensive. Our intent is solely to contribute to the understanding and detection of harmful language in online environments.

---


## 📬 Contact 

**Email:**
- [oscar.toimil@rai.usc.es](mailto:oscar.toimil@rai.usc.es)
- [marcosfernandez.pichel@usc.es](marcosfernandez.pichel@usc.es)
- [ezra.aragon@usc.es](ezra.aragon@usc.es)








