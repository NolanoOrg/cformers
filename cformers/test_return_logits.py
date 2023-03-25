"""Call's the C++ code from Python for returning logits."""
from subprocess import Popen, PIPE
import re
import sys
import torch
import transformers as tf # RIP TensorFlow

def get_cpp_logits(model_type, model_path, prompt):
    """Generates text from the given prompt."""
    # By now prompt should be a list of integers, sanity check this once
    assert isinstance(prompt, list), f"Prompt should be a list of integers: {prompt}"
    assert all([isinstance(x, int) for x in prompt]), \
        f"Prompt should be a list of integers {prompt}"

    command = ["./cpp/main", model_type, "-m", model_path,
               "--prompt", " ".join([str(x) for x in prompt]),
               "--return_logits"]
    print(" ".join(command))

    process = Popen(command, stdout=PIPE, stderr=PIPE)
    all_stdout = ""
    # Read all the output from the process reading one byte at a time
    for chr in iter(lambda: process.stdout.read(1), b""):
        all_stdout += chr.decode('utf-8')

    # Get the logits enclosed between '\n<|BEGIN>' and ' <END|>\n' (possibly multiline)
    # Use regex over r'\n<\|BEGIN\>.*<END\|\>'
    logits_str_match = re.search(r'\n<\|BEGIN\>.*<END\|\>', all_stdout, re.DOTALL)
    logits_str = all_stdout[logits_str_match.start():logits_str_match.end()]

    # Convert the logits to a list of list of floats
    logits_str = logits_str.replace('<|BEGIN>', '').replace('<END|>', '').strip()
    logits_str_lines = [x for x in logits_str.splitlines() if x.strip()]

    # Sanity check that either line begins with "logits:"
    for line in logits_str_lines:
        if not line.strip().startswith("logits:"):
            print(line[:10])
            return (None, process.wait(), logits_str_match, all_stdout)
        # assert line.strip().startswith("logits:"), f"Line {line} does not begin with 'logits:'"
    # Convert to a tensor, ignoring the "logits:" prefix
    logit_tensor = torch.tensor([[float(x) for x in line.strip().split()[1:]]
                                 for line in logits_str_lines])

    # Wait for the process to finish and return its exit code
    return (logit_tensor, process.wait(), logits_str_match, all_stdout)

def get_tf_logits(model_card, prompt):
    """Generates text from the given prompt."""
    # By now prompt should be a list of integers, sanity check this once
    assert isinstance(prompt, list), f"Prompt should be a list of integers: {prompt}"
    assert all([isinstance(x, int) for x in prompt]), \
        f"Prompt should be a list of integers {prompt}"

    # Load the model
    model = tf.AutoModelForCausalLM.from_pretrained(model_card)

    # Generate the logits
    with torch.no_grad():
        logits = model(torch.LongTensor([prompt]))[0] # pylint: disable=no-member

    return logits

def get_logits(model_card, model_type, model_path, prompt):
    """Returns logits from both C++ and Python."""
    # First get_cpp_logits, then get_tf_logits
    cpp_logits = get_cpp_logits(model_type, model_path, prompt)
    print("Obtained logits from C++")
    tf_logits = get_tf_logits(model_card, prompt)
    print("Obtained logits from Python")
    return cpp_logits, tf_logits
# c, t = get_logits("EleutherAI/pythia-160m-deduped", "gptneox",
#                   "../models/pythia-160m-deduped/ggml-model-pythia-160m-deduped-f32.bin",
#                   [1, 2, 3, 4, 5])

if __name__ == "__main__":
    arg_model_card = sys.argv[1]
    arg_model_type = sys.argv[2]
    arg_model_path = sys.argv[3]
    dummy_inputs = [1, 2, 3, 4, 5]

    cpp_logits, tf_logits = get_logits(
        arg_model_card, arg_model_type, arg_model_path, dummy_inputs)
    print("C++ logits:", cpp_logits)
    print("Python logits:", tf_logits)
