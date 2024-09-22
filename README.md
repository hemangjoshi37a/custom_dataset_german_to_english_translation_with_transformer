

# Custom Dataset German to English Translation with Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models%20on%20HuggingFace-blue)](https://huggingface.co/hemangjoshi37a/german_to_english_hjlabsin/tree/main)

## Project Overview

This project, developed at the Technion - Israel Institute of Technology, focuses on creating a custom dataset and implementing a Transformer model for German to English translation. The goal is to achieve high-quality translations with a minimum BLEU score of 35% on the validation set.

## Features

- Custom dataset creation for German to English translation
- Implementation of a Transformer model for translation tasks
- Performance evaluation using BLEU score
- Handling of root words and modifiers in unlabeled data

## Project Requirements

- Achieve a minimum average BLEU score of 35% on the validation set
- Implement a tagger that processes files in `comp.unlabeled` format
- Use Transformer architecture for the model
- Train on `train.labeled` file only
- Evaluate performance on `val.labeled` file
- Generate translations for `comp.unlabeled` file

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/custom_dataset_german_to_english_translation_with_transformer.git
   cd custom_dataset_german_to_english_translation_with_transformer
   ```

2. Set up the Python environment (Python 3.8 required):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

## Usage

1. Training the model:
   ```
   python train_model.py
   ```

2. Evaluating on the validation set:
   ```
   python evaluate_model.py --file val.unlabeled
   ```

3. Generating translations for the competition file:
   ```
   python generate_comp_tagged.py --input comp.unlabeled --output comp_id1_id2.labeled
   ```

## Results

### Untrained Model
![Untrained Model Results](https://user-images.githubusercontent.com/12392345/229338317-525ccf12-7b37-45bf-a971-a012bd583b7c.png)

### After Training
![Trained Model Results](https://user-images.githubusercontent.com/12392345/229338315-14ffc44a-1877-4141-8178-eb22858c1caa.png)

## Pre-trained Models

You can download the pre-trained models from our [HuggingFace repository](https://huggingface.co/hemangjoshi37a/german_to_english_hjlabsin/tree/main).

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Hemang Joshi - [hemangjoshi37a@gmail.com](mailto:hemangjoshi37a@gmail.com)

Project Link: [https://github.com/your-username/custom_dataset_german_to_english_translation_with_transformer](https://github.com/your-username/custom_dataset_german_to_english_translation_with_transformer)

## Acknowledgements

- [Technion - Israel Institute of Technology](https://www.technion.ac.il/en/home-2/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [BLEU Score Implementation](https://github.com/mjpost/sacrebleu)


## 📫 How to reach me
[<img height="36" src="https://cdn.simpleicons.org/WhatsApp"/>](https://wa.me/917016525813) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/telegram"/>](https://t.me/hjlabs) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Gmail"/>](mailto:hemangjoshi37a@gmail.com) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/LinkedIn"/>](https://www.linkedin.com/in/hemang-joshi-046746aa) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/facebook"/>](https://www.facebook.com/hemangjoshi37) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Twitter"/>](https://twitter.com/HemangJ81509525) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/tumblr"/>](https://www.tumblr.com/blog/hemangjoshi37a-blog) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/StackOverflow"/>](https://stackoverflow.com/users/8090050/hemang-joshi) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Instagram"/>](https://www.instagram.com/hemangjoshi37) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Pinterest"/>](https://in.pinterest.com/hemangjoshi37a) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Blogger"/>](http://hemangjoshi.blogspot.com) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/similarweb"/>](https://hjlabs.in/) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/gitlab"/>](https://gitlab.com/hemangjoshi37a) &nbsp;



