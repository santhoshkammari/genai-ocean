# model.generate() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_ids` | torch.LongTensor | Required | The input sequence of token ids. Shape: (batch_size, sequence_length) |
| `max_length` | int | None | The maximum length of the sequence to be generated. |
| `max_new_tokens` | int | None | The maximum number of new tokens to generate, ignoring the number of tokens in the prompt. |
| `min_length` | int | None | The minimum length of the sequence to be generated. |
| `do_sample` | bool | False | Whether or not to use sampling; use greedy decoding otherwise. |
| `early_stopping` | bool | False | Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not. |
| `num_beams` | int | 1 | Number of beams for beam search. 1 means no beam search. |
| `temperature` | float | 1.0 | The value used to module the next token probabilities. |
| `top_k` | int | 50 | The number of highest probability vocabulary tokens to keep for top-k-filtering. |
| `top_p` | float | 1.0 | If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept for generation. |
| `repetition_penalty` | float | 1.0 | The parameter for repetition penalty. 1.0 means no penalty. |
| `length_penalty` | float | 1.0 | Exponential penalty to the length. 1.0 means no penalty. |
| `no_repeat_ngram_size` | int | 0 | If set to int > 0, all ngrams of that size can only occur once. |
| `encoder_no_repeat_ngram_size` | int | 0 | If set to int > 0, all ngrams of that size that occur in the encoder input can not occur in the decoder output. |
| `bad_words_ids` | List[List[int]] | None | List of token ids that are not allowed to be generated. |
| `force_words_ids` | List[List[int]] or List[List[List[int]]] | None | List of token ids that must be generated. |
| `num_return_sequences` | int | 1 | The number of independently computed returned sequences for each element in the batch. |
| `max_time` | float | None | The maximum amount of time you allow the computation to run for in seconds. |
| `attention_mask` | torch.LongTensor | None | Mask to avoid performing attention on padding token indices. Shape: (batch_size, sequence_length) |
| `decoder_start_token_id` | int | None | If an encoder-decoder model starts decoding with a different token than BOS. |
| `use_cache` | bool | None | Whether or not the model should use the past last key/values attentions to speed up decoding. |
| `num_beam_groups` | int | 1 | Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams. |
| `diversity_penalty` | float | 0.0 | This value is subtracted from a beam's score if it generates a token same as any beam from other group at a particular time. |
| `prefix_allowed_tokens_fn` | Callable[[int, torch.Tensor], List[int]] | None | If provided, this function constraints the beam search to allowed tokens only at each step. |
| `logits_processor` | LogitsProcessorList | None | Custom logits processors that complement the default logits processors built from arguments and a model's config. |
| `renormalize_logits` | bool | False | Whether to renormalize the logits after applying all the logits processors or warpers. |
| `stopping_criteria` | StoppingCriteriaList | None | Custom stopping criteria that complement the default stopping criteria built from arguments and a model's config. |
| `constraints` | List[Constraint] | None | Custom constraints that can be added to the generation to ensure that the output will contain the use of certain tokens as defined by `Constraint` objects. |
| `output_attentions` | bool | False | Whether or not to return the attentions tensors of all attention layers. |
| `output_hidden_states` | bool | False | Whether or not to return the hidden states of all layers. |
| `output_scores` | bool | False | Whether or not to return the prediction scores. |
| `return_dict_in_generate` | bool | False | Whether or not to return a `ModelOutput` instead of a plain tuple. |
| `forced_bos_token_id` | int | None | The id of the token to force as the first generated token after the decoder_start_token_id. |
| `forced_eos_token_id` | Union[int, List[int]] | None | The id of the token to force as the last generated token when `max_length` is reached. |
| `remove_invalid_values` | bool | None | Whether to remove possible nan and inf outputs of the model to prevent the generation method to crash. |
| `exponential_decay_length_penalty` | Tuple[Union[int, float]] | None | This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been generated. |
| `suppress_tokens` | List[int] | None | A list of tokens that will be suppressed at generation. |
| `begin_suppress_tokens` | List[int] | None | A list of tokens that will be suppressed at the beginning of the generation. |
| `forced_decoder_ids` | List[List[int]] | None | A list of pairs of integers which indicates a mapping from generation indices to token indices that will be forced before sampling. |
| `sequence_bias` | Dict[Tuple[int], float] | None | Dictionary that maps a sequence of tokens to its bias term. |
| `guidance_scale` | float | None | The guidance scale for classifier free guidance (CFG). |

Note: Not all models support all parameters. The exact set of supported parameters can vary depending on the specific model and its configuration. Always refer to the documentation of the specific model you're using for the most accurate information.
