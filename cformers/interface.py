"""Call's the C++ code from Python."""
from subprocess import Popen, PIPE
import hashlib
import re
import os
import sys
import select
import wget

import transformers as tf # RIP TensorFlow

sys.path.append("./cpp/")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# Get CFORMERS_CACHE_PATH if it exists as an environment variable else use the default
if "CFORMERS_CACHE_PATH" in os.environ:
    CFORMERS_CACHE_PATH = os.environ["CFORMERS_CACHE_PATH"]
else:
    CFORMERS_CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cformers")

class ModelUrlMap:
    """Stores the URL mapping for various models."""
    def __init__(self,
                 cpp_model_name,
                 int4_fixed_zero="",
                 int4_variable_zero="",
                 gptq_int4_fixed_zero="",
                 gptq_int4_variable_zero=""):
        """Constructor for ModelUrlMap"""
        self.cpp_model_name = cpp_model_name
        self.int4_fixed_zero = int4_fixed_zero
        self.int4_variable_zero = int4_variable_zero
        self.gptq_int4_fixed_zero = gptq_int4_fixed_zero
        self.gptq_int4_variable_zero = gptq_int4_variable_zero

    def get_url(self, mode):
        """Returns the URL for the given mode."""
        url = None
        if mode == "int4_fixed_zero":
            url = self.int4_fixed_zero
        elif mode == "int4_variable_zero":
            url = self.int4_variable_zero
        elif mode == "gptq_int4_fixed_zero":
            url = self.gptq_int4_fixed_zero
        elif mode == "gptq_int4_variable_zero":
            url = self.gptq_int4_variable_zero
        else:
            raise ValueError("Invalid mode: {}".format(mode))

        if url == "":
            raise ValueError("{} not available for this model, please choose from: {}".format(mode, self.get_modes()))
        return url
        
    def get_modes(self):
        return [
            mode for mode_str, mode in [("int4_fixed_zero", self.int4_fixed_zero),
                                        ("int4_variable_zero", self.int4_variable_zero),
                                        ("gptq_int4_fixed_zero", self.gptq_int4_fixed_zero),
                                        ("gptq_int4_variable_zero", self.gptq_int4_variable_zero)]
            if mode_str != ""]

MAP_MODEL_TO_URL = { # Replace "/" with "-.-" in the model name
    # GPT-J
    'EleutherAI/gpt-j-6B': ModelUrlMap(
        cpp_model_name="gptj",
        int4_fixed_zero="https://huggingface.co/ayushk4/EleutherAI-.-gpt-j-6B/resolve/main/int4_fixed_zero.bin"),
    'Salesforce/codegen-2B-mono': ModelUrlMap(
        cpp_model_name="gptj",
        int4_fixed_zero="https://huggingface.co/ayushk4/Salesforce-.-codegen-2B-mono/resolve/main/int4-fixed-zero.bin"),
    'Salesforce/codegen-6B-mono': ModelUrlMap(
        cpp_model_name="gptj",
        int4_fixed_zero="https://huggingface.co/ayushk4/Salesforce-.-codegen-6B-mono/resolve/main/int4-fixed-zero.bin"),
    'Salesforce/codegen-16B-mono': ModelUrlMap(
        cpp_model_name="gptj",
        int4_fixed_zero="https://huggingface.co/kamalojasv/Salesforce-.-codegen-16B-mono/resolve/main/int4-fixed-zero"),

    # Bloom
    'bigscience/bloom-560m': ModelUrlMap(
        cpp_model_name="bloom",
        int4_fixed_zero="https://huggingface.co/tejasvaidhya/bloom-560m-4bit-quant.bin/resolve/main/int4_fixed_zero.bin"),
    'bigscience/bloom-1b1': ModelUrlMap(
        cpp_model_name="bloom",
        int4_fixed_zero="https://huggingface.co/tejasvaidhya/bloom-1b1-4bit-quant.bin/resolve/main/int4_fixed_zero.bin"),
    'bigscience/bloom-1b7': ModelUrlMap(
        cpp_model_name="bloom",
        int4_fixed_zero="https://huggingface.co/tejasvaidhya/bloom-1b7-4bit-quant.bin/resolve/main/int4_fixed_zero.bin"),
    'bigscience/bloom-3b': ModelUrlMap(
        cpp_model_name="bloom",
        int4_fixed_zero="https://huggingface.co/tejasvaidhya/bloom-3b-4bit-quant.bin/resolve/main/int4_fixed_zero.bin"),
    'bigscience/bloom-7b1': ModelUrlMap(
        cpp_model_name="bloom",
        int4_fixed_zero="https://huggingface.co/ayushk4/bigscience-.-bloom-7b1/resolve/main/int4_fixed_zero.bin"),
    
    # GPT2
    'gpt2': ModelUrlMap(
        cpp_model_name="gpt2",
        int4_fixed_zero="https://huggingface.co/kamalojasv/gpt2/resolve/main/int4_fixed_zero"),

    # togethercomputer/OpenChatKit
    'togethercomputer/GPT-NeoXT-Chat-Base-20B': ModelUrlMap(
        cpp_model_name="gptneox",
        int4_fixed_zero="https://huggingface.co/Black-Engineer/OpenChatKit_q4/resolve/main/int4_fixed_zero"),

    # GPT-NeoX based
    'OpenAssistant/oasst-sft-1-pythia-12b': ModelUrlMap(
        cpp_model_name="gptneox",
        int4_fixed_zero="https://huggingface.co/ayushk4/OpenAssistant-.-oasst-sft-1-pythia-12b/resolve/main/int4_fixed_zero.bin"),
}

class AutoInference:
    """A wrapper for the C++ model."""
    def __init__(self, model_name, hash_sum="", mode="int4_fixed_zero"):
        self.model_name = model_name
        self.mode = mode
        self.hash_sum = hash_sum
        self.cpp_model_name = MAP_MODEL_TO_URL[model_name].cpp_model_name
        self.model_url = MAP_MODEL_TO_URL[model_name].get_url(mode)
        self.model_save_path = os.path.join(CFORMERS_CACHE_PATH, "models", model_name, mode)
        self.tokenizer = tf.AutoTokenizer.from_pretrained(model_name)

        # Download the model if it doesn't exist
        if not os.path.exists(self.model_save_path):
            # Create the directory if it doesn't exist
            parent_dir = os.path.dirname(self.model_save_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            print("Downloading model...")
            def bar_progress(current, total, width=80):
                progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
                sys.stdout.write("\r" + progress_message)
                sys.stdout.flush()
            wget.download(self.model_url, self.model_save_path, bar=bar_progress)

            print("Download complete!")
        # TODO: Check the hash sum of the downloaded model

    def generate(self,
                 prompt,
                 top_k=20,
                 top_p=0.95,
                 temperature=00.85,
                 num_tokens_to_generate=10,
                 repeat_last_n=64,
                 repeat_penalty=1.3,
                 n_threads=8,
                 seed=42,
                 streaming_token_str_hook=lambda x: x,
                 streaming_token_ids_hook=lambda x: x,
                 print_streaming_output=True):
        """Generates text from the given prompt.

        streaming_output_hook: function to be called after every token is generated.
        """
        if isinstance(prompt, str):
            # Tokenize and get the input ids
            prompt = self.tokenizer.encode_plus(prompt)['input_ids']
        # By now prompt should be a list of integers, sanity check this once
        assert isinstance(prompt, list), f"Prompt should be a list of integers: {prompt}"
        assert all([isinstance(x, int) for x in prompt]), \
            f"Prompt should be a list of integers {prompt}"
        # Convert to a string of space separated integers
        prompt = " ".join([str(x) for x in prompt])

        command = ["./cpp/main", self.cpp_model_name,
                   "-m", self.model_save_path,
                   "--prompt", prompt,
                   "--seed", str(seed),
                   "--threads", str(n_threads),
                   "--n_predict", str(num_tokens_to_generate),
                   "--top_k", str(top_k),
                   "--top_p", str(top_p),
                   "--temp", str(temperature),
                   "--repeat_last_n", str(repeat_last_n),
                   "--repeat_penalty", str(repeat_penalty)]
        print(" ".join(command))

        process = Popen(command, stdout=PIPE, stderr=PIPE)
        tokens_ids_so_far = []
        has_generation_begun = False
        token_id_buffer = ""
        all_stdout_so_far = ""
        for c in iter(lambda: process.stdout.read(1), b""):
            all_stdout_so_far += c.decode('utf-8')

            if not has_generation_begun:
                to_print = c.decode('utf-8')
            else:
                if ' ' in c.decode('utf-8') and token_id_buffer.strip():
                    # We have a token id
                    token_id = int(token_id_buffer.strip())
                    token_str = self.tokenizer.decode([token_id])
                    token_id_buffer = ""
                    tokens_ids_so_far.append(token_id)
                    # Call the streaming output hooks
                    streaming_token_str_hook(token_str)
                    streaming_token_ids_hook(token_id)
                    to_print = token_str
                else:
                    token_id_buffer += c.decode('utf-8')

            if print_streaming_output and to_print:
                print(to_print, end='')
                to_print = ""
                sys.stdout.flush()

            if '<|BEGIN> ' in all_stdout_so_far:
                has_generation_begun = True

            # Check if the line is empty or matches the end marker
            if '<END|>' in all_stdout_so_far:
                if print_streaming_output:
                    print("\n---------------------\n")
                break

            # # Also check for errors
            # err = process.stderr.readline().decode('utf-8').strip()
            # if err:
            #     raise Exception(err)

        # print('\n' + '-'*30, all_stdout_so_far, '-'*30 + '\n')
        # return all_stdout_so_far
        token_line = re.findall(r'<\|BEGIN\>(.*?)<END\|>', all_stdout_so_far, re.DOTALL)[0]

        # Convert the token_line to a list of integers
        all_tokens = [int(x) for x in token_line.split()]

        # Decode the tokens
        decoded_tokens = self.tokenizer.decode(all_tokens)

        # Get the exit code
        success = process.wait()
        # Kill the child process if it's still running
        if process.poll() is None:
            process.kill()
            # wait for the process to terminate
            process.wait()

        # Wait for the process to finish and return its exit code
        return {"success": success,
                "token_ids": all_tokens,
                "token_str": decoded_tokens}
        
