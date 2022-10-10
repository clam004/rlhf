import collections

import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import top_k_top_p_filtering

WANDB_PADDING = -1

def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def flatten_dict(nested, sep='/'):
    """Flatten dictionary and concatenate nested keys with separator."""
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.abc.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v
    flat = {}
    rec(nested, '', flat)
    return flat

def stack_dicts(stats_dicts):
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:
        stats_list = [torch.flatten(d[k]) for d in stats_dicts]
        results[k] = pad_sequence(stats_list, batch_first=True, padding_value=WANDB_PADDING)
    return results

def add_suffix(input_dict, suffix):
    """Add suffix to dict keys."""
    return dict((k + suffix, v) for k,v in input_dict.items())


def pad_to_size(tensor, size, dim=1, padding=50256):
    """Pad tensor to size."""
    t_size = tensor.size()[dim]
    if t_size==size:
        return tensor
    else:
        return torch.nn.functional.pad(tensor, (0,size-t_size), 'constant', padding)

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def whiten(values, shift_mean=True):
    """Whiten values."""
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped

def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd*logits, axis=-1)
    return entropy


def average_torch_dicts(list_of_dicts):
    """Average values of a list of dicts wiht torch tensors."""
    average_dict = dict()
    for key in list_of_dicts[0].keys():
        average_dict[key] = torch.mean(torch.stack([d[key] for d in list_of_dicts]), axis=0)
    return average_dict

def stats_to_np(stats_dict):
    """Cast all torch.tensors in dict to numpy arrays."""
    new_dict = dict()
    for k, v in stats_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.detach().cpu().numpy()
        else:
            new_dict[k] = v
        if np.isscalar(new_dict[k]):
            new_dict[k] = float(new_dict[k])
    return new_dict

def listify_batch(tensor):
    """Turns the first dimension of a tensor into a list."""
    return [tensor[i] for i in range(tensor.shape[0])]


def build_bert_batch_from_txt(text_list, tokenizer, device):
    """Create token id and attention mask tensors from text list for BERT classification."""
    
    # tokenize
    tensors = [tokenizer.encode(txt, return_tensors="pt").to(device) for txt in text_list]
    
    # find max length to pad to
    max_len = max([t.size()[1] for t in tensors])
    
    # get padded tensors and attention masks
    # (attention masks make bert ignore padding)
    padded_tensors = []
    attention_masks = []
    for tensor in tensors:
        attention_mask = torch.ones(tensor.size(), device=device)
        padded_tensors.append(pad_to_size(tensor, max_len, padding=0))
        attention_masks.append(pad_to_size(attention_mask, max_len, padding=0))
    
    # stack all tensors
    padded_tensors = torch.cat(padded_tensors)
    attention_masks = torch.cat(attention_masks)  
    
    return padded_tensors, attention_masks


class LengthSampler:
    """
    """
    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))
    def __call__(self):
        return np.random.choice(self.values)


def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0):
    """Sample text from language model."""
    input_ids = queries
    for i in range(txt_len):
        # Get Logits
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return input_ids[:, -txt_len:]


def convo_list_dic2list_str(
  conversation_list_dic,
  human_symbol = '[H]: ',
  bot_symbol = '[B]: ',
  utterance_delimiter = '\n',
):

  """ This function takes a list of dictionaries
  and turns them into a list of speaker_symbol + utterance strings

  Args: 
      conversation_list_dic (List[Dict]): 
      ie: [{'speaker': 'bot', 'utterance': 'im waking up!'},
           {'speaker': 'human', 'utterance': 'wakey wakey sleepyhead'}, ...]

  Returns:
      conversation_list_str (List[str]): list of speaker_symbol + utterance strings
      ie: ['\n[C]: Hello Fara.','\n[A]: Hello! How are you doing today?',...]
  """

  speaker2symbol = {
      'bot':bot_symbol,
      'human':human_symbol,
  }

  conversation_list_str = list()

  for u in conversation_list_dic:

      speaker_symbol = speaker2symbol[u['speaker']]
      utterance = end_punctuation(u['utterance'])

      conversation_list_str.append(utterance_delimiter + speaker_symbol + utterance)

  # Elicit next agent utterance
  conversation_list_str.append(utterance_delimiter + bot_symbol)

  return conversation_list_str


def generate_extract_replies(
    model,
    tokenizer,
    prompt,
    max_gen_len = 16, 
    no_repeat_ngram_size = None,
    pad_token_id = 50256,
    do_sample = True,
    top_k = 100, 
    top_p = 0.99, 
    num_return_sequences = 1,
    temperature = 0.9,
    stop_strings = [
        '<',
        '[human]',
        '\n',
        '[',
    ],
    verbose = False,
):

    ''' This function predicts the next utterance
    in a conversation
    '''

    gen_texts = generate_text(
        model,
        tokenizer,
        prompt,
        max_gen_len = max_gen_len, 
        no_repeat_ngram_size = no_repeat_ngram_size,
        pad_token_id = pad_token_id,
        do_sample = do_sample,
        top_k = top_k, 
        top_p = top_p, 
        num_return_sequences = num_return_sequences,
        temperature = temperature,
        verbose = verbose,
    )

    replies = [
        extract_str(
            gen_text,
            prefix = prompt,
            stop_strings = stop_strings,
            verbose = verbose,
        )
        for gen_text in gen_texts
    ]

    return replies


def generate_text(
    model,
    tokenizer,
    prompt,
    max_gen_len = 16, 
    no_repeat_ngram_size = None,
    pad_token_id = 50256,
    do_sample = True,
    top_k = 100, 
    top_p = 0.99, 
    num_return_sequences = 1,
    temperature = 0.9,
    verbose = False,
):

    ''' function for generating text from an input into 
    the app.package model

    prompt (str): text to be tokenized and pushed through model

    if you are doing few shot detection you should leave 
    no_repeat_ngram_size = None and max_len = 16
    as long as the default max_len is more than the expected
    label text

    we leave it up to the label extractor to clip of the portion
    of the generated text that you need
    '''
    NUM_GPUS = torch.cuda.device_count()

    prompt_dic = tokenizer(prompt,return_tensors="pt")
    prompt_ids = prompt_dic.input_ids
    prompt_mask = prompt_dic.attention_mask
    prompt_len = prompt_ids.shape[1]

    if verbose:
        print('prompt_ids.shape', prompt_ids.shape)
        print('prompt_mask.shape', prompt_mask.shape)

    if NUM_GPUS > 0:
        prompt_ids = prompt_ids.to(model.device)
        prompt_mask = prompt_mask.to(model.device)

    output_ids = model.generate(
        prompt_ids,
        attention_mask = prompt_mask,
        max_length = prompt_len + max_gen_len,
        no_repeat_ngram_size = no_repeat_ngram_size,
        pad_token_id = pad_token_id,
        do_sample = do_sample,
        top_k = top_k, 
        top_p = top_p, 
        num_return_sequences = num_return_sequences,
        temperature = temperature,
    )

    generated_text = tokenizer.batch_decode(output_ids)

    return generated_text