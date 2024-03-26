required data are in augi1 server

/local_datasets/llm/archive/

config.json

gemma-7b-it-quant.ckpt

tokenizer.model

<br/><br/>

 $ cd /local_datasets/llm
 
 $ ls
 
 archive
 
 ForumMessages.csv
 
<br/><br/>


ForumMessages.csv include original_texts

archive folder includecheckpoint from pretrained gemma model

<br/><br/>

**modify the code**

original.py ,line 14

sys.path.append("/your working directory/gemma_pytorch/")

<br/><br/>

**run the code**
sbatch llm.sh

it runs original.py
