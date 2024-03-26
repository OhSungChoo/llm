import pandas as pd 
forum_messages_df = pd.read_csv('/local_datasets/llm/ForumMessages.csv')
forum_messages_df.head()

original_texts=forum_messages_df['Message'][:5]

rewrite_prompts = [
    'Explain this to me like I\'m five.',
    'Convert this into a sea shanty.',
    'Make this rhyme.',
]

import sys 
sys.path.append("/data/ohsung/repos/llm/gemma_pytorch/") 
from gemma.config import GemmaConfig, get_config_for_7b, get_config_for_2b
from gemma.model import GemmaForCausalLM
from gemma.tokenizer import Tokenizer
import contextlib
import os
import torch

# Load the model
VARIANT = "7b-it-quant" 
MACHINE_TYPE = "cuda" 
weights_dir = '/local_datasets/llm/archive' 

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
  """Sets the default torch dtype to the given dtype."""
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(torch.float)

# Model Config.
model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
model_config.tokenizer = os.path.join(weights_dir, "tokenizer.model")
model_config.quant = "quant" in VARIANT

# Model.
device = torch.device(MACHINE_TYPE)
with _set_default_tensor_type(model_config.get_dtype()):
  model = GemmaForCausalLM(model_config)
  ckpt_path = os.path.join(weights_dir, f'gemma-{VARIANT}.ckpt')
  model.load_weights(ckpt_path)
  model = model.to(device).eval()


import random
random.seed(0)
# This is the prompt format the model expects
USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

rewrite_data = []

for original_text in original_texts:
    rewrite_prompt = rewrite_prompts[0]
    prompt = f'{rewrite_prompt}\n{original_text}'
    rewritten_text = model.generate(
        USER_CHAT_TEMPLATE.format(prompt=prompt),
        device=device,
        output_len=100,
    )
    rewrite_data.append({
        'original_text': original_text,
        'rewrite_prompt': rewrite_prompt,
        'rewritten_text': rewritten_text,
    })


    
# Let's turn our generated data into a dataframe, and spot check the first rewrite to see if it makes sense.
rewrite_data_df = pd.DataFrame(rewrite_data)
rewrite_data_df[:5].values
print(rewrite_data_df[:5].values)