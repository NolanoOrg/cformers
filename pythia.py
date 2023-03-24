import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompt", nargs="+",help="Enter a value for the prompt")
parser.add_argument("-t", "--tokens",help="Number of tokens to generate",type=int, default=100)
args = parser.parse_args()

from interface import AutoInference as AI
ai = AI('OpenAssistant/oasst-sft-1-pythia-12b')

if not args.prompt:
    while True:
        name = input("Please enter your prompt (type 'exit' to quit): ")
        if name.lower() == 'exit':
            break
        x = ai.generate("<|prompter|>"+str(name)+"<|endoftext|><|assistant|>", num_tokens_to_generate=args.tokens)
        print(x['token_str'])
else:
    my_prompt = ' '.join(args.prompt)
    x = ai.generate("<|prompter|>"+my_prompt+"<|endoftext|><|assistant|>", num_tokens_to_generate=args.tokens)
    print(x['token_str'])

