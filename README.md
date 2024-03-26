required data are in augi1 server

/local_datasets/llm/

/local_datasets/llm/archive/

config.json

gemma-7b-it-quant.ckpt

tokenizer.model

 $ cd /local_datsets/llm
 
 $ ls
 
 archive
 ForumMessages.csv
 

ForumMessages.csv include original_texts
archive folder includecheckpoint from pretrained gemma model

**modify the code**

original.py ,line 14

sys.path.append("/your working directory/gemma_pytorch/")


