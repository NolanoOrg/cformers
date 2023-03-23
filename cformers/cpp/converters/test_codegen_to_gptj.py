import os, wget, json
import torch, transformers as tf

dummy_input_ids = torch.randint(0, 51200, (1, 512), dtype=torch.long)
tokenizer = tf.AutoTokenizer.from_pretrained('Salesforce/codegen-6B-mono')
ip = tokenizer.batch_encode_plus(["def parse_html(html):"], return_tensors="pt")

codegen_gptj = tf.AutoModelForCausalLM.from_pretrained("codegen/", low_cpu_mem_usage=True)
codegen = tf.AutoModelForCausalLM.from_pretrained('Salesforce/codegen-6B-mono')

# Set the nn.moduleList codegen_gptj.transformer.h to first element of codegen.transformer.h
codegen_gptj.transformer.h = torch.nn.ModuleList(**codegen_gptj.transformer.h[:1])
# same for codegen
codegen.transformer.h = torch.nn.ModuleList(**codegen.transformer.h[:1])

op1 = codegen(**ip).logits
op2 = codegen_gptj(**ip).logits
