import os
import sys
import struct
import json
import torch
import wget
import numpy as np

from transformers import GPTJForCausalLM, CodeGenForCausalLM
print(sys.argv)

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1)) + [ord(" "), ord("\t")]
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

if len(sys.argv) < 4:
    print("Usage: convert-h5-to-ggml.py model-card modeldir [use-f32]\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)

# output in the same directory as the model
model_card = sys.argv[1]
fname_out = sys.argv[2] + "/ggml-gptj-6b-model.bin"

if "CONVERTER_CACHE_DIR" in os.environ:
    dir_cache = os.environ["CONVERTER_CACHE_DIR"] + model_card.replace('/', '-.-')
else:
    dir_cache = "~/.cformers_converters" + model_card.replace('/', '-.-')

if not os.path.exists(dir_cache):
    os.makedirs(dir_cache)

# Fetch vocab.json from https://huggingface.co/<model_card>/resolve/main/vocab.json if not found in dir_cache/vocab.json
if not os.path.exists(dir_cache + "/vocab.json"):
    print("Downloading vocab.json from " + model_card)
    wget.download("https://huggingface.co/" + model_card + "/resolve/main/vocab.json",
                  dir_cache + "/vocab.json")
if not os.path.exists(dir_cache + "/added_tokens.json"):
    print("Downloading added_tokens.json from " + model_card)
    wget.download("https://huggingface.co/" + model_card + "/resolve/main/added_tokens.json",
                  dir_cache + "/added_tokens.json")
if not os.path.exists(dir_cache + "/config.json"):
    print("Downloading config.json from " + model_card)
    wget.download("https://huggingface.co/" + model_card + "/resolve/main/config.json",
                  dir_cache + "/config.json")

with open(dir_cache + "/vocab.json", "r") as f:
    encoder = json.load(f)

with open(dir_cache + "/added_tokens.json", "r") as f:
    encoder_added = json.load(f)

with open(dir_cache + "/config.json", "r") as f:
    hparams = json.load(f)

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

# if len(sys.argv) > 2:
ftype = int(sys.argv[3])
if ftype < 0 or ftype > 1:
    print("Invalid ftype: " + str(ftype))
    sys.exit(1)
fname_out = sys.argv[2] + "/ggml-" + model_card.replace('/', '-.-') + "-" + \
            ftype_str[ftype] + ".bin"

is_codegen = False
if "gptj" in hparams['architectures'][0].lower():
    model = GPTJForCausalLM.from_pretrained(model_card, low_cpu_mem_usage=True)
elif "codegen" in hparams['architectures'][0].lower():
    is_codegen = True
    model = CodeGenForCausalLM.from_pretrained(model_card, low_cpu_mem_usage=True)
else:
    raise Exception("Unknown model type: " + str(
        hparams['architectures']) + " for model_card: " + model_card)
#print (model)

list_vars = model.state_dict()
#print (list_vars)

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["vocab_size"]))
# fout.write(struct.pack("i", hparams["n_positions"]))
fout.write(struct.pack("i", hparams["n_embd"]))
fout.write(struct.pack("i", hparams["n_head"]))
fout.write(struct.pack("i", hparams["n_layer"]))
fout.write(struct.pack("i", hparams["rotary_dim"]))
fout.write(struct.pack("i", ftype))

byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}
# ['ł', 'Ń']
byte_decoder['ł'] = byte_decoder['Ł']
byte_decoder['Ń'] = byte_decoder['N']

if is_codegen:
    # We need to add dummy values to encder_added to make len(encoder) + len(encoder_added) == hparams["vocab_size"]
    for i in range(hparams["vocab_size"] - (len(encoder) + len(encoder_added))):
        encoder_added["<dummy" + str(i) + ">"] = len(encoder) + len(encoder_added)

fout.write(struct.pack("i", len(encoder) + len(encoder_added)))

for key in encoder:
    text = bytearray([byte_decoder[c] for c in key])
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

for key in encoder_added:
    text = bytearray([byte_decoder[c] for c in key])
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

if is_codegen:
    new_list_vars = {}
    # for each transformer.h.<LAYER_ID>.attn.qkv_proj.weight we need to split it into 3 separate weights
    for name in list_vars.keys():
        if name.endswith("attn.qkv_proj.weight"):
            data = list_vars[name]
            n_dims = len(data.shape)
            assert n_dims == 2
            n_embd = hparams["n_embd"]
            assert n_embd * 3 == data.shape[0], f"n_embd * 3 != data.shape[0] ({n_embd} * 3 != {data.shape[0]})"
            q_unshaped, v_unshaped, k_unshaped = torch.split(
                data.reshape(4, -1, n_embd), n_embd//4, dim=1)
            q_shaped, v_shaped, k_shaped = (
                q_unshaped.reshape(-1, n_embd), v_unshaped.reshape(-1, n_embd), k_unshaped.reshape(-1, n_embd))
            new_list_vars[name.replace(".qkv_proj.", ".q_proj.")] = q_shaped
            new_list_vars[name.replace(".qkv_proj.", ".v_proj.")] = v_shaped
            new_list_vars[name.replace(".qkv_proj.", ".k_proj.")] = k_shaped
            # sanity check equivalence over identity matrix
            hidden_state = torch.eye(n_embd, dtype=torch.float32)
            qkv = torch.nn.Linear(n_embd, n_embd * 3, bias=False)
            qkv.weight = torch.nn.Parameter(data)
            q = torch.nn.Linear(n_embd, n_embd, bias=False)
            q.weight = torch.nn.Parameter(q_shaped)
            k = torch.nn.Linear(n_embd, n_embd, bias=False)
            k.weight = torch.nn.Parameter(k_shaped)
            v = torch.nn.Linear(n_embd, n_embd, bias=False)
            v.weight = torch.nn.Parameter(v_shaped)
            def pass_codegen(qkv_proj, hidden_states): # from https://github.com/huggingface/transformers/blob/main/src/transformers/models/codegen/modeling_codegen.py
                def _split_heads_codegen(x, n_head, dim_head, mp_num):
                    reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
                    reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
                    return reshaped
                qkv = qkv_proj(hidden_states)
                mp_num = 4
                head_dim = model.config.n_embd // model.config.n_head
                qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))
                local_dim = model.config.n_head * head_dim // mp_num
                query, value, key = torch.split(qkv_split, local_dim, dim=-1)
                query = _split_heads_codegen(query, model.config.n_head, head_dim, mp_num=mp_num)
                key = _split_heads_codegen(key, model.config.n_head, head_dim, mp_num=mp_num)
                value = _split_heads_codegen(value, model.config.n_head, head_dim, mp_num=mp_num)
                value = value.permute(0, 2, 1, 3)
                return query, key, value
            def pass_gptj(q_proj, k_proj, v_proj, hidden_states): # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_gptj.py
                def _split_heads_gptj(tensor, num_attention_heads, attn_head_size, rotary):
                    new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
                    tensor = tensor.view(new_shape)
                    if rotary:
                        return tensor
                    if len(tensor.shape) == 5:
                        return tensor.permute(0, 1, 3, 2, 4)  # (batch, blocks, head, block_length, head_features)
                    elif len(tensor.shape) == 4:
                        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
                    else:
                        raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
                query, key, value = (q_proj(hidden_states), k_proj(hidden_states), v_proj(hidden_states))
                n_head, head_dim = model.config.n_head, model.config.n_embd // model.config.n_head
                query, key, value = (_split_heads_gptj(query, n_head, head_dim, True),
                                    _split_heads_gptj(key, n_head, head_dim, True),
                                    _split_heads_gptj(value, n_head, head_dim, False))
                return query, key, value
            def do_sanity_check(q, k, v, qkv):
                hidden_state = torch.eye(n_embd, dtype=torch.float32).unsqueeze(0)
                q1, k1, v1 = pass_gptj(q, k, v, hidden_state)
                q2, k2, v2 = pass_codegen(qkv, hidden_state)
                assert torch.allclose(q1, q2), ("q1: " + str(q1) + " q2: " + str(q2))
                assert torch.allclose(k1, k2), ("k1: " + str(k1) + " k2: " + str(k2))
                assert torch.allclose(v1, v2), ("v1: " + str(v1) + " v2: " + str(v2))
                print(f"Sanity check passed! for {name}")
            qkv = torch.nn.Linear(n_embd, 3 * n_embd, bias=False)
            qkv.weight = torch.nn.Parameter(data)
            # print("qkv: ", qkv, "q: ", q, "k: ", k, "v: ", v)
            # print("qkv.weigh.shape", qkv.weight.shape, "q.weight.shape", q.weight.shape, "k.weight.shape", k.weight.shape, "v.weight.shape", v.weight.shape)
            do_sanity_check(q, k, v, qkv)
        elif name.endswith("attn.causal_mask"):
            new_list_vars[name.replace('.causal_mask', '.bias')] = list_vars[name]
            new_list_vars[name.replace('.causal_mask', '.masked_bias')] = torch.tensor(-1e9)
        else:
            new_list_vars[name] = list_vars[name]
    list_vars = new_list_vars

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()

    # we don't need these
    if name.endswith("attn.masked_bias") or name.endswith(".attn.bias") or name.endswith(".attn.causal_mask"):
        print("  Skipping variable: " + name)
        continue
    print("Processing variable: " + name + " with shape: ", data.shape, data[:2, :2] if len(data.shape) == 2 else data[:2])

    n_dims = len(data.shape)

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0
    if ftype != 0:
        if name[-7:] == ".weight" and n_dims == 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0
    else:
        if data.dtype != np.float32:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

    # for efficiency - transpose these matrices:
    # (note - with latest ggml this is no longer more efficient, so disabling it)
    #  "transformer.h.*.mlp.fc_in.weight"
    #  "transformer.h.*.attn.out_proj.weight"
    #  "transformer.h.*.attn.q_proj.weight"
    #  "transformer.h.*.attn.k_proj.weight"
    #  "transformer.h.*.attn.v_proj.weight"
    #if name.endswith(".mlp.fc_in.weight")     or \
    #   name.endswith(".attn.out_proj.weight") or \
    #   name.endswith(".attn.q_proj.weight")   or \
    #   name.endswith(".attn.k_proj.weight")   or \
    #   name.endswith(".attn.v_proj.weight"):
    #    print("  Transposing")
    #    data = data.transpose()

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str)

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")

if is_codegen:
    print("Also saving as pytorch gptj model")
    # make `codegen/{model_card}` directory if it doesn't exist
    if not os.path.exists(f"codegen/{model_card}"):
        os.makedirs(f"codegen/{model_card}")
    torch.save(list_vars, f"codegen/{model_card}/pytorch_model.bin")
    config = {
        "activation_function": "gelu_new",
        "architectures": [
            "GPTJForCausalLM"
        ],
        "attn_pdrop": 0.0,
        "bos_token_id": 1,
        "embd_pdrop": 0.0,
        "eos_token_id": 50256,
        "gradient_checkpointing": False,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gptj",
        "n_embd": model.config.n_embd,
        "n_head": model.config.n_head,
        "n_inner": None,
        "n_layer": model.config.n_layer,
        "n_positions": 2048,
        "resid_pdrop": 0.0,
        "rotary": True,
        "rotary_dim": model.config.rotary_dim,
        "scale_attn_weights": True,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "task_specific_params": {
            "text-generation": {
            "do_sample": True,
            "max_length": 50,
            "temperature": 1.0
            }
        },
        "tie_word_embeddings": False,
        "tokenizer_class": "GPT2Tokenizer",
        "transformers_version": "4.21.0.dev0",
        "use_cache": True,
        "vocab_size": 51200
        }
    # save config at `codegen/{model_card}/config.json`
    json.dump(config, open(f"codegen/{model_card}/config.json", "w"))
