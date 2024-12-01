# Project Overview

This project involves the utilization of state-of-the-art language models to enhance text generation processes. Below are the primary models used and their respective download links:

1. **CHATGLM3**: An advanced generative language model developed by THUDM. 
   - Download: [Hugging Face - chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k/tree/main)
   
2. **BLOOM**: A large-scale language model by BigScience.
   - Download: [Hugging Face - bloom-7b1](https://huggingface.co/bigscience/bloom-7b1/tree/main)
   
3. **SciBERT**: A BERT model pre-trained on scientific text.
   - Download: [GitHub - allenai/scibert](https://github.com/allenai/scibert)

## Data Location
The original data has been uploaded to this [Google Drive](https://drive.google.com/drive/folders/1AXRWmhfG0CJ-VxyIhxro4jIMfiSvmddj?usp=sharing) because it is large. 
You can download it to `/data` directory.

## Scripts Execution
The following Python scripts are used to generate and train models using the specified data and phases.

### Hard Attention Generation
Run `hard_atten_generation.py` to generate hard attention data:
```bash
python hard_atten_generation.py --phase chatglm3-6b-32k --data dblp
```

### Information Generation
Run `information_generate.py` to process information generation:
```bash
python information_generate.py --phase chatglm3-6b-32k --data dblp
```

# Model Training and Prediction
Use `main.py` to train models and predict outcomes under different configurations:
1. RAHA
```bash
python main.py --phase train --model chatglm3-6b-32k --data dblp --epoch 5 --update --adapter
```
2. Model without Hard Attention
```bash
python main.py --phase train --model chatglm3-6b-32k --data dblp --epoch 5 --ori
```
3. Model without Adapter
```bash
python main.py --phase train --model chatglm3-6b-32k --data dblp --epoch 5 --update
```
4. Model without Recurrent Alignment
```bash
python main.py --phase train --model chatglm3-6b-32k --data dblp --epoch 5 --adapter
```