import argparse
from interface import AutoInference as AI
model_map = {'pythia': 'OpenAssistant/oasst-sft-1-pythia-12b', 'bloom': 'bigscience/bloom-7b1', 'gptj': 'EleutherAI/gpt-j-6B'}

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompt", nargs="+",help="Enter a value for the prompt")
parser.add_argument("-t", "--tokens",help="Number of tokens to generate",type=int, default=100)
parser.add_argument("-m", "--model", help="Specify a Model", choices=model_map.keys(),default="pythia")
args = parser.parse_args()

def generate(prompt,arg=args):
    print('Model is '+arg.model)
    if arg.model == 'pythia':
        x = ai.generate("<|prompter|>"+prompt+"<|endoftext|><|assistant|>", num_tokens_to_generate=arg.tokens)
    elif arg.model == 'bloom':
        x = ai.generate(""+prompt+"", num_tokens_to_generate=arg.tokens)
    elif arg.model == 'gptj':
        x = ai.generate(""+prompt+"", num_tokens_to_generate=arg.tokens)
    else:
        x = ai.generate("<|prompter|>"+prompt+"<|endoftext|><|assistant|>", num_tokens_to_generate=arg.tokens)
    return x

ai = AI(model_map[args.model])

if not args.prompt:
    while True:
        my_prompt = input("Please enter your prompt (type 'exit' to quit): ")
        if my_prompt.lower() == 'exit':
            break
        x =generate(my_prompt,args)
        print(x['token_str'])
else:
    my_prompt = ' '.join(args.prompt)
    x =generate( my_prompt,args)
    print(x['token_str'])
