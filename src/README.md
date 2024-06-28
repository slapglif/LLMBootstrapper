## 🌑 LLMBootstrapper: Your Gateway to Downstream LLMs 

Welcome to LLMBootstrapper! 🥾 This project aims to streamline your journey from "I want to make an LLM" to "I've got a working downstream task model!" 🏆

**Tired of:**

* 😫  Struggling to find the right configuration for training?
* 😵  Spending hours on boilerplate code and data processing?
* 🤯  Getting lost in a maze of confusing libraries and dependencies?

**LLMBootstrapper is here to help!** 🎉 

**What LLMBootstrapper Provides:**

* **⚡️ Ready-to-use code:** Get started quickly with a well-structured foundation for your project. 
* **📚  Comprehensive documentation:**  Understand the code and concepts with detailed explanations and comments. 
* **💪  Best practices:**  Benefit from common techniques and tips gathered from experience. 
* **🚀  Flexibility:**  Customize and extend the framework to fit your specific requirements.

**Here's a sneak peek at the core features:**

| Feature | Description |
|---|---|
| **Data Loading & Preprocessing** |  Streamlined data loading and processing for common datasets like WikiText, Squad, Codeparrot, etc.  |
| **Model Training & Evaluation** |  PyTorch Lightning-based training and evaluation, leveraging mixed precision, early stopping, and more.  |
| **Uncertainty Estimation** |  Integrate uncertainty estimation techniques like Monte Carlo Dropout and Temperature Scaling.  |
| **Data Augmentation** |  Apply text and token augmentation methods for improved robustness and generalization.  |
| **Benchmarking** |  Run comprehensive benchmarks for language modeling, text generation, and uncertainty estimation.  |
| **Visualization** |  Generate insightful plots for perplexity, ROUGE scores, uncertainty metrics, and more.  |


**Getting Started** 

1. **Clone the repository:** 
   ```bash
   git clone https://github.com/your-username/LLMBootstrapper.git
   ```
2. **Install dependencies:** 
   ```bash
   pip install -r requirements.txt 
   ```
3. **Choose your dataset:**
   - `wikitext`
   - `squad`
   - `codeparrot`
   - `winogrande`
   - `medical_questions_pairs`
   - `misc` (for general language modeling on WikiText-2)
4. **Configure your training parameters:**
   - Edit the `config.json` file to customize hyperparameters and dataset paths.
5. **Implement Your LightningModule:**
   -  LLMBootstrapper provides the framework, but you'll need to implement your own `LightningModule`  which will inherit from `pl.LightningModule`.
     *  This module should contain your model architecture, forward pass, loss function, and optimizer.
     *  It also includes hooks for training, validation, and testing.
   - We assume you're working with a language modeling task, so ensure your model has an LM Head (e.g., a linear layer with `vocab_size` outputs).
   - You'll need to use a tokenizer with a strong vocabulary suitable for your dataset (e.g., `transformers.AutoTokenizer`). 
6. **Run the main script:**
   ```bash
   python engine/main.py --config config.json --mode train --dataset <your_dataset> --cache_dir <your_cache_dir>
   ```

**Key Considerations:**

* **Model Architecture:** LLMBootstrapper works with any PyTorch model. However, many of the assumptions are made for a language modeling task. 
* **Tokenizer:**  You'll need to select a tokenizer with a strong vocabulary suited for your task and dataset. 
* **LM Head:** Your model should have a language modeling head for generating text.

**Future Improvements:**

In upcoming releases, LLMBootstrapper will offer the ability to automatically import and initialize pretrained models directly from the command line.  

**Happy Hacking! 💻**

This project is still under development, but I'm excited to share it with you. If it saves you even one hour of time
copy and pasting some of my blocks to have a semi-working training method you didnt have to hack at, that would make me happy.

**License:** MIT License

**Note:** This project is designed to be used as a starting point for building downstream task LLMs. It does not include a complete implementation for all tasks or methods. You can extend and customize the framework based on your specific needs. 



