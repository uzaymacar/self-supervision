import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class DataCollator(object):
    def __init__(self):
        pass

    def __call__(self, examples):
        pass

    def _preprocess_batch(self, examples):
        """
        Preprocesses tensors: returns the current examples in case all the tensors are the same
        length, otherwise pads tensors with [PAD] until they are the same length. This function
        is made adaptable to both work with (i) tuple of tensors as is the case when __call__() is
        used as the collate_fn() function of a torch.utils.data.DataLoader() and with (ii) tensor
        with shape (B, ...) where B = batch size (i.e. pre-tensorized)
        """
        if all(x.size(0) == examples[0].size(0) for x in examples):
            if isinstance(examples, tuple):
                return torch.stack(examples, dim=0)
            elif isinstance(examples, torch.Tensor):
                return examples
            else:
                raise ValueError('The type of examples "%s" is not recognized!' % str(type(examples)))
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError('Pad token not found in @tokenizer!')
            elif isinstance(examples, tuple):
                return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            elif isinstance(examples, torch.Tensor):
                return pad_sequence([example for example in examples], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            else:
                raise ValueError('The type of examples "%s" is not recognized!' % str(type(examples)))


class MLMCollator(DataCollator):
    """
    Data collator used for masked language modeling (MLM). It collates batches of tensors, honoring
    their tokenizer's pad_token and preprocesses batches with choosing each token with a
    @mask_probability probability, and replaces 80% of these with [MASK], 10% with a random token
    from the vocabulary, and leaves the 10% as is by default in BERT. To play around with this
    behaviour, check the @policy parameter.

    :param (BertTokenizer) tokenizer: tokenizer for the logic expressions with encode() and decode()
    :param (float) mask_probability: Probability used for choosing whether a token is to be included
           or not in the MLM task.
    :param (int) nomask_id: integer IDs to indicate tokens that shouldn't be masked
    :param (None or list(float)) policy: list where the first index is the probability of
           replacement by [MASK], second index is the probability of replacement by random token,
           and third index is the probability of leaving as is. The three values should sum
           up to 1. When left empty (i.e. None), it adopts the classic BERT masking strategy.
    :param (None or list(int)) special_token_ids: list of integer IDs of tokens that are 'special';
           these tokens will not be masked and will not be replaced to ensure integrity of data.
           NOTE: Special tokens w.r.t. to the tokenizer are already taken care of thanks to the
           tokenizer.get_special_tokens_mask() function; this parameter gives a chance to extend it.
    """
    def __init__(self, tokenizer, mask_probability, nomask_id=-100, policy=None, special_token_ids=None):
        super(MLMCollator).__init__()

        if tokenizer.mask_token is None:
            raise ValueError(' Tokenizer is missing the mask token (e.g. [MASK], <mask>, etc.)')

        self.tokenizer = tokenizer
        self.mask_probability = mask_probability

        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.nomask_id = nomask_id

        self.policy = policy or [0.8, 0.1, 0.1]
        if sum(self.policy) != 1.0:
            raise ValueError('The elements of @policy should sum up to 1.')

        self.special_token_ids = special_token_ids or [self.tokenizer.bos_token_id,
                                                       self.tokenizer.eos_token_id,
                                                       self.tokenizer.sep_token_id,
                                                       self.tokenizer.pad_token_id,
                                                       self.tokenizer.cls_token_id]

    def __call__(self, examples):
        """
        Makes the class callable, just like a function. This is following the new 'transformers'
        convention. In the old versions, this would instead be a standard collate_batch() function.
        """
        examples_ = examples.clone()
        batch = self._preprocess_batch(examples_)
        input_ids, labels = self.mask_tokens(input_ids=batch)
        return input_ids, labels

    def mask_tokens(self, input_ids):
        """
        Prepares masked tokens input and label pairs for MLM: 80% MASK, 10% random, 10% original.
        The @size arguments in torch functions used here (e.g. torch.full(), torch.randint())
        make this function work with a batch of input IDs. Moreover, functions like
        get_special_tokens_mask() work for a single input ID, but is called in a for loop over the
        batch.

        :param (torch.Tensor) input_ids: tensor with shape (B, S), contains sequences
        :return: (torch.Tensor, torch.Tensor) input_ids, labels
        """
        # In masked language modeling, the labels are the input itself (i.e. self-supervision)
        labels = input_ids.clone()

        # We sample a few tokens in each sequence for masked-LM training with mask probability
        probabilities = torch.full(size=labels.shape, fill_value=self.mask_probability)
        # Get masks for special tokens (e.g. [CLS] and [SEP] for BertTokenizer)
        special_tokens_mask = []
        for labels_ in labels.tolist():
            mask = self.tokenizer.get_special_tokens_mask(labels_, already_has_special_tokens=True)
            special_tokens_mask.append(mask)
        # Get a boolean vector where T is for indices where special tokens are, otherwise F
        special_indices = torch.tensor(data=special_tokens_mask, dtype=torch.bool)
        # Fill in special token indices with 0.0 - we don't want them masked
        probabilities.masked_fill_(special_indices, value=0.0)

        # Check if certain other token (e.g. [PAD], [UNK], etc.) exists in the tokenizer
        for special_token_id in self.special_token_ids:
            if special_token_id:
                # Get the related indices in the input IDs
                special_indices = labels.eq(special_token_id)
                # Fill in special indices with 0.0 - we don't want them masked
                probabilities.masked_fill_(mask=special_indices, value=0.0)

        # Get masked indices with a Bernoulli distribution based on p = probabilities
        masked_indices = torch.bernoulli(probabilities).bool()
        # Set everything except the masked indices to some large, negative number
        labels[~masked_indices] = self.nomask_id

        # 80% (default) of the time, we replace masked input tokens with the mask token
        replaced_indices = torch.bernoulli(torch.full(size=labels.shape, fill_value=self.policy[0])).bool()
        replaced_indices = replaced_indices & masked_indices
        input_ids[replaced_indices] = self.mask_id

        # 10% (default) of the time, we replace the remaining masked input tokens with random tokens
        # NOTE: Fill value is 0.5, but the remaining pct is %20 -> 0.2 x 0.5 = 0.10 (default)
        randomized_indices = torch.bernoulli(torch.full(size=labels.shape, fill_value=self.policy[1]/(1-self.policy[0]))).bool()
        randomized_indices = randomized_indices & masked_indices & ~replaced_indices
        random_tokens = torch.randint(high=len(self.tokenizer), size=labels.shape, dtype=torch.long)

        # Remove special tokens from random tokens (e.g. [SEP] may create a problem in some cases)
        allowed_token_ids = [i for i in range(len(self.tokenizer)) if i not in self.special_token_ids]
        for special_token_id in self.special_token_ids:
            replacement_token_id = np.random.choice(allowed_token_ids, size=1)
            random_tokens[random_tokens == special_token_id] = torch.tensor(replacement_token_id, dtype=torch.long)

        input_ids[randomized_indices] = random_tokens[randomized_indices]

        # 10% (default) of the time, we keep the remaining masked input tokens unchanged
        return input_ids, labels


class SequentialMLMCollator(DataCollator):
    """
    Data collator used for sequential masked language modeling (sMLM). It collates batches of
    tensors, honoring their tokenizer's pad_token and preprocesses batches. Check the docstring
    of self.sequentially_mask_tokens() to get more information.

    :param (BertTokenizer) tokenizer: tokenizer for the logic expressions with encode() and decode()
    :param (int) nomask_id: integer IDs to indicate tokens that shouldn't be masked
    :param (None or list(int)) special_token_ids: list of integer IDs of tokens that are 'special';
           these tokens will not be masked and will not be replaced to ensure integrity of data.
           NOTE: Special tokens w.r.t. to the tokenizer are already taken care of thanks to the
           tokenizer.get_special_tokens_mask() function; this parameter gives a chance to extend it.
    """
    def __init__(self, tokenizer, nomask_id=-100, special_token_ids=None):
        super(SequentialMLMCollator).__init__()

        if tokenizer.mask_token is None:
            raise ValueError(' Tokenizer is missing the mask token (e.g. [MASK], <mask>, etc.)')

        self.tokenizer = tokenizer

        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.nomask_id = nomask_id

        self.special_token_ids = special_token_ids or [self.tokenizer.bos_token_id,
                                                       self.tokenizer.eos_token_id,
                                                       self.tokenizer.sep_token_id,
                                                       self.tokenizer.pad_token_id,
                                                       self.tokenizer.cls_token_id]

    def __call__(self, examples):
        """
        Makes the class callable, just like a function. This is following the new 'transformers'
        convention. In the old versions, this would instead be a standard collate_batch() function.
        """
        examples_ = examples.clone()
        batch = self._preprocess_batch(examples_)
        input_ids, labels = self.sequentially_mask_tokens(input_ids=batch)
        return input_ids, labels

    def sequentially_mask_tokens(self, input_ids, exclude_tokens=None):
        """
        Prepares masked tokens input and label pairs for MLM: given an input IDs of shape
        (B=1, S) (NOTE: The batch size should be 1), it sequentially creates a list of length S'
        of masked tokens input and label pairs, masking one token at a single time, going L-to-R.
        It is much like the standard L-to-R language modeling task except you have a greater,
        bidirectional context window. Such a task can be used when a complete reconstruction
        loss is to be calculated over a single input (e.g. anomaly detection).

        :param (torch.Tensor) input_ids: tensor with shape (B=1, S), contains sequences
        :param (list) exclude_tokens: list of tokens that should be not be masked, in addition to
               the @special_token_ids matched tokens.
        :return: (torch.Tensor, torch.Tensor) input_ids, labels
        """
        # This function only works with a single example batch at a time
        if input_ids.shape[0] != 1:
            raise ValueError('Function sequentially_mask_tokens() only works with batch size = 1!')

        # Get token IDs to be excluded
        exclude_ids = self.tokenizer.convert_tokens_to_ids(exclude_tokens) if exclude_tokens else []
        # In masked language modeling, the labels are the input itself (i.e. self-supervision)
        labels = input_ids.clone()

        for special_token_id in self.special_token_ids:
            # If a certain token (e.g. [PAD], [SEP], etc.) exists in the tokenizer
            if special_token_id:
                # Get the related indices in the input IDs
                special_indices = labels.eq(special_token_id)
                # Set special token (e.g. [CLS], [SEP]) indices to some large, negative number
                labels[special_indices] = self.nomask_id

        # Create a list of input IDs where each next example masks the next available index
        batch_input_ids, batch_labels = [], []
        # NOTE: We are momentarily converting labels to a one-dimensional vector of the form (S),
        #       from the previous (B=1,S) so that we can iterate over the actual input IDs
        for index, input_id in enumerate(labels[0, :].tolist()):
            if input_id != self.nomask_id and input_id not in exclude_ids:
                current_input_ids, current_labels = input_ids.clone(), labels.clone()
                current_input_ids[:, index] = self.mask_id
                current_labels[:, 0:index] = self.nomask_id
                current_labels[:, index+1:] = self.nomask_id
                batch_input_ids.append(current_input_ids.view(-1))
                batch_labels.append(current_labels.view(-1))

        # Tensorize batch input IDs and labels
        return torch.stack(batch_input_ids, dim=0), torch.stack(batch_labels, dim=0)


class PermutationMLMCollator(DataCollator):
    """
    Data collator used for permutation-based masked language modeling (pMLM). It collates batches of
    tensors, honoring their tokenizer's pad_token and preprocesses batches with a two step algorithm
    where spans of tokens are [MASK]ed and then a permutation of sequence length is applied to get
    a factorization order as per XLNet.
    """
    def __init__(self, tokenizer, mask_probability, max_span_length=5, nomask_id=-100, special_token_ids=None):
        super(PermutationMLMCollator).__init__()

        if tokenizer.mask_token is None:
            raise ValueError(' Tokenizer is missing the mask token (e.g. [MASK], <mask>, etc.)')

        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.max_span_length = max_span_length

        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.nomask_id = nomask_id

        self.special_token_ids = special_token_ids or [self.tokenizer.bos_token_id,
                                                       self.tokenizer.eos_token_id,
                                                       self.tokenizer.sep_token_id,
                                                       self.tokenizer.pad_token_id,
                                                       self.tokenizer.cls_token_id]

    def __call__(self, examples):
        batch = self._preprocess_batch(examples)
        inputs, permutation_mask, labels = self.permutate_and_mask_tokens(batch)
        return inputs, permutation_mask, labels

    def permutate_and_mask_tokens(self, input_ids):
        """
        Permutates and masks tokens. It uses an algorithm with two steps.
        Step 1: Iterate over the sequence and mark spans of tokens with [MASK].
        Step 2: Generate permutation indices, i.e. sample a random factorisation order for sequence
                which will determine which tokens a given token can attend to.

        In Step 2, the logic for whether the i-th token can attend on the j-th token is based on the
        factorization order:
            (a) 0, CAN ATTEND: perm_index[i] > perm_index[j] OR j is neither masked nor functional
            (b) 1, CAN'T ATTEND: perm_index[i] <= perm_index[j] AND j is either masked or functional

        NOTE: Functional tokens are [SEP], [CLS], [PAD], etc. (i.e. those indicated by
              self.special_token_ids. Non-functional tokens are all of the remaining tokens.

        :param (torch.Tensor) input_ids: tensor with shape (B, S), contains sequences
        :return: (torch.Tensor, torch.Tensor, torch.Tensor) input_ids, permutation_mask, labels
        """
        # In masked language modeling, the labels are the input itself (i.e. self-supervision)
        labels = input_ids.clone()
        # Initialize the masked indices with all zeros
        masked_indices = torch.full(labels.shape, fill_value=0.0, dtype=torch.bool)

        # Do for each example in batch
        for i in range(labels.shape[0]):
            # Start from the beginning of the sequence; j is the num. of tokens processed
            j = 0
            while j < labels.shape[1]:
                # Sample a span_length (i.e. length of span of tokens to be masked)
                span_length = torch.randint(1, self.max_span_length + 1, (1,)).item()
                # Reserve a context of length to surround the span to be masked
                context_length = int(span_length / self.mask_probability)
                # Sample a starting point and mask tokens in the computed span
                start_index = j + torch.randint(context_length - span_length + 1, (1,)).item()
                if start_index >= labels.shape[1]:
                    break
                masked_indices[i, start_index: start_index + span_length] = 1
                # Update num. tokens processed
                j += context_length

        # Store all of the special indices, those set by self.special_token_ids, in a mask
        all_special_indices = torch.zeros(labels.shape).bool()
        # Check if certain other token (e.g. [PAD], [UNK], etc.) exists in the tokenizer
        for special_token_id in self.special_token_ids:
            if special_token_id:
                # Get the related indices in the input IDs
                special_indices = labels.eq(special_token_id)
                all_special_indices |= special_indices
                # Fill in special indices with 0.0 - we don't want them masked
                masked_indices.masked_fill_(mask=special_indices, value=0.0)

        # Mask the selected indices in input IDs and and set everything else to a large, negative
        # number in labels to ensure that loss is not computed on them
        input_ids[masked_indices] = self.tokenizer.mask_token_id
        labels[~masked_indices] = self.nomask_id

        # Initialize permutation mask with shape (B, S, S)
        permutation_mask = torch.zeros((labels.shape[0], labels.shape[1], labels.shape[1]), dtype=torch.float32)

        # Do for each example in batch
        for i in range(labels.shape[0]):
            # Create a linear factorisation order
            permutation_index = torch.randperm(n=labels.shape[1])
            # Set the permutation indices of non-masked (non-functional) tokens to -1 s.t.
            # (1) They can be seen by all other positions
            # (2) They cannot see masked positions, so there won't be information leak
            permutation_index.masked_fill_(~masked_indices[i] & ~all_special_indices[i], -1)

            # To match the shape of masked_indices, let's reshape the permutation index
            # NOTE: This is so that it is compatible with the upcoming logic
            permutation_index = permutation_index.reshape((labels.shape[1], 1))

            # Apply the attention attending logic based on the factorization order
            permutation_mask[i] = (permutation_index <= permutation_index) & masked_indices[i]
            # NOTE: If permutation_mask[k, i, j] = 0, then i can attend to j in batch k;
            #       If permutation_mask[k, i, j] = 1, then i can NOT attend to j in batch k.

        return input_ids, permutation_mask, labels


class TextInfillingCollator(DataCollator):
    """
    Data collator used for text infilling masked language modeling (MLM). It collates batches of
    tensors, honoring their tokenizer's pad_token and preprocesses batches with [MASK]ing spans
    of tokens.

    :param (BertTokenizer) tokenizer: tokenizer for the logic expressions with encode() and decode()
    :param (float) mask_probability: Probability used for choosing whether a token is to be included
           or not in the MLM task.
    :param (int) nomask_id: integer IDs to indicate tokens that shouldn't be masked
    :param (int) mean_span_length: expected rate, lambda, in the Poisson distribution for sampling
           span lengths for each of the spans in each sequence
    :param (str) mask_strategy: if set to 'spanbert', replaces each span with a sequence of [MASK]
           tokens of exactly the same length, or if set to 'bart', replaces each span with a single
           [MASK] token.
    :param (None or list(int)) special_token_ids: list of integer IDs of tokens that are 'special';
           these tokens will not be masked and will not be replaced to ensure integrity of data.
           NOTE: Special tokens w.r.t. to the tokenizer are already taken care of thanks to the
           tokenizer.get_special_tokens_mask() function; this parameter gives a chance to extend it.
    """
    def __init__(self, tokenizer, mask_probability, mean_span_length,
                 mask_strategy='spanbert', nomask_id=-100, special_token_ids=None):
        super(TextInfillingCollator).__init__()

        if tokenizer.mask_token is None:
            raise ValueError(' Tokenizer is missing the mask token (e.g. [MASK], <mask>, etc.)')

        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.mean_span_length = mean_span_length

        if mask_strategy not in ['spanbert', 'bart']:
            raise ValueError('Masking strategy "%s" is not recognized!' % mask_strategy)
        self.mask_strategy = mask_strategy

        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.nomask_id = nomask_id

        self.special_token_ids = special_token_ids or [self.tokenizer.bos_token_id,
                                                       self.tokenizer.eos_token_id,
                                                       self.tokenizer.sep_token_id,
                                                       self.tokenizer.pad_token_id,
                                                       self.tokenizer.cls_token_id]

    def __call__(self, examples):
        """
        Makes the class callable, just like a function. This is following the new 'transformers'
        convention. In the old versions, this would instead be a standard collate_batch() function.
        """
        examples_ = examples.clone()
        batch = self._preprocess_batch(examples_)
        input_ids, labels = self.infill_and_mask_tokens(input_ids=batch)
        return input_ids, labels

    def infill_and_mask_tokens(self, input_ids):
        """
        Prepares masked tokens input and label pairs for text-infilled MLM.

        :param (torch.Tensor) input_ids: tensor with shape (B, S), contains sequences
        :return: (torch.Tensor, torch.Tensor) input_ids, labels
        """
        # In masked language modeling, the labels are the input itself (i.e. self-supervision)
        labels = input_ids.clone()
        # Initialize the masked indices with all zeros
        masked_indices = torch.full(labels.shape, fill_value=0.0, dtype=torch.bool)

        # Do for each example in batch
        for i in range(labels.shape[0]):
            # Start from the beginning of the sequence; j is the num. of tokens processed
            j = 0
            while j < labels.shape[1]:
                # Sample a span_length (i.e. length of span of tokens to be masked)
                span_length = torch.poisson(torch.tensor(self.mean_span_length, dtype=torch.float)).long().item()
                if span_length >= labels.shape[1]:
                    span_length = self.mean_span_length
                # Reserve a context of length to surround the span to be masked
                context_length = int(span_length / self.mask_probability)
                # Sample a starting point and mask tokens in the computed span
                start_index = j + torch.randint(context_length-span_length + 1, (1,)).item()
                if start_index >= labels.shape[1]:
                    break
                # Simply replace all tokens in the span with [MASK] tokens if 'spanbert'
                if self.mask_strategy == 'spanbert':
                    masked_indices[i, start_index: start_index+span_length] = 1
                # Replace all tokens in the span with a single [MASK] token and add n-1 [PAD] tokens
                elif self.mask_strategy == 'bart':
                    masked_indices[i, start_index] = 1
                    input_ids[i, start_index+1:] = torch.cat(
                        [input_ids[i, start_index+span_length:],
                         torch.full(size=(span_length-1, ), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)]
                    )
                # Update num. tokens processed
                j += context_length

        # Store all of the special indices, those set by self.special_token_ids, in a mask
        all_special_indices = torch.zeros(labels.shape).bool()
        # Check if certain other token (e.g. [PAD], [UNK], etc.) exists in the tokenizer
        for special_token_id in self.special_token_ids:
            if special_token_id:
                # Get the related indices in the input IDs
                special_indices = labels.eq(special_token_id)
                all_special_indices |= special_indices
                # Fill in special indices with 0.0 - we don't want them masked
                masked_indices.masked_fill_(mask=special_indices, value=0.0)

        # Mask the selected indices in input IDs and and set everything else to a large, negative
        # number in labels to ensure that loss is not computed on them
        input_ids[masked_indices] = self.tokenizer.mask_token_id
        if self.mask_strategy == 'spanbert':
            labels[~masked_indices] = self.nomask_id
        elif self.mask_strategy == 'bart':
            labels[all_special_indices] = self.nomask_id

        return input_ids, labels


class TokenDeletionCollator(DataCollator):
    """
    Data collator used for token deletion masked language modeling (MLM). It collates batches of
    tensors, honoring their tokenizer's pad_token and preprocesses batches with [MASK]ing tokens
    and removing these tokens. In contrast to default token masking as shown in MLMCollator(), the
    model must decide which poistions are missing inputs.

    :param (BertTokenizer) tokenizer: tokenizer for the logic expressions with encode() and decode()
    :param (float) mask_probability: Probability used for choosing whether a token is to be included
            or not in the MLM task.
    :param (int) nomask_id: integer IDs to indicate tokens that shouldn't be masked
    :param (None or list(int)) special_token_ids: list of integer IDs of tokens that are 'special';
            these tokens will not be masked and will not be replaced to ensure integrity of data.
            NOTE: Special tokens w.r.t. to the tokenizer are already taken care of thanks to the
            tokenizer.get_special_tokens_mask() function; this parameter gives a chance to extend it.
    """
    def __init__(self, tokenizer, mask_probability, nomask_id=-100, special_token_ids=None):
        super(TokenDeletionCollator).__init__()

        if tokenizer.mask_token is None:
            raise ValueError(' Tokenizer is missing the mask token (e.g. [MASK], <mask>, etc.)')

        self.tokenizer = tokenizer
        self.mask_probability = mask_probability

        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.nomask_id = nomask_id

        self.special_token_ids = special_token_ids or [self.tokenizer.bos_token_id,
                                                       self.tokenizer.eos_token_id,
                                                       self.tokenizer.sep_token_id,
                                                       self.tokenizer.pad_token_id,
                                                       self.tokenizer.cls_token_id]

    def __call__(self, examples):
        """
        Makes the class callable, just like a function. This is following the new 'transformers'
        convention. In the old versions, this would instead be a standard collate_batch() function.
        """
        examples_ = examples.clone()
        batch = self._preprocess_batch(examples_)
        input_ids, labels = self.delete_tokens(input_ids=batch)
        return input_ids, labels

    def delete_tokens(self, input_ids):
        """
        Prepares reduced tokens input and label pairs for token deletion pretext task.

        :param (torch.Tensor) input_ids: tensor with shape (B, S), contains sequences
        :return: (torch.Tensor, torch.Tensor) input_ids, labels
        """
        # In masked language modeling, the labels are the input itself (i.e. self-supervision)
        labels = input_ids.clone()

        # We sample a few tokens in each sequence for masked-LM training with mask probability
        probabilities = torch.full(size=labels.shape, fill_value=self.mask_probability)

        # Store all of the special indices, those set by self.special_token_ids, in a mask
        all_special_indices = torch.zeros(labels.shape).bool()
        # Check if certain other token (e.g. [PAD], [UNK], etc.) exists in the tokenizer
        for special_token_id in self.special_token_ids:
            if special_token_id:
                # Get the related indices in the input IDs
                special_indices = labels.eq(special_token_id)
                all_special_indices |= special_indices
                probabilities.masked_fill_(special_indices, value=0.0)

        # Get masked indices with a Bernoulli distribution based on p = probabilities
        masked_indices = torch.bernoulli(probabilities).bool()

        # Do for each example in batch
        for i in range(labels.shape[0]):
            # Get num. deletions based on whether token has been marked with [MASK] or not
            num_deletions = len(masked_indices[i, :][masked_indices[i, :]])
            # Restructure input s.t. [MASK]ed tokens removed, and same num. [PAD]s added at the end
            input_ids[i, :] = torch.cat(
                [input_ids[i, :][~masked_indices[i, :]],
                 torch.full(size=(num_deletions, ), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)]
            )

        # Everything except the special tokens should be visible in this task
        # TODO: Is this a correct approach? Can we do something smarter, and more efficient?
        #       Another approach would be to return the indices of deleted tokens!
        labels[all_special_indices] = self.nomask_id

        return input_ids, labels


class DocumentRotationCollator(DataCollator):
    """
    Data collator used for document rotation pretext task. It collates batches of tensors, honoring
    their tokenizer's pad_token and preprocesses batches by uniformly choosing a token and rotating
    each batch such that it starts with the respective token. Hence, the model learns to identify
    the start of the document.

    :param (BertTokenizer) tokenizer: tokenizer for the logic expressions with encode() and decode()
    :param (int) nomask_id: integer IDs to indicate tokens that shouldn't be masked
    :param (None or list(int)) special_token_ids: list of integer IDs of tokens that are 'special';
            these tokens will not be masked and will not be replaced to ensure integrity of data.
            NOTE: Special tokens w.r.t. to the tokenizer are already taken care of thanks to the
            tokenizer.get_special_tokens_mask() function; this parameter gives a chance to extend it.
    """
    def __init__(self, tokenizer, nomask_id=-100, special_token_ids=None):
        super(DocumentRotationCollator).__init__()

        if tokenizer.mask_token is None:
            raise ValueError(' Tokenizer is missing the mask token (e.g. [MASK], <mask>, etc.)')

        self.tokenizer = tokenizer
        self.nomask_id = nomask_id

        self.special_token_ids = special_token_ids or [self.tokenizer.bos_token_id,
                                                       self.tokenizer.eos_token_id,
                                                       self.tokenizer.sep_token_id,
                                                       self.tokenizer.pad_token_id,
                                                       self.tokenizer.cls_token_id]

    def __call__(self, examples):
        """
        Makes the class callable, just like a function. This is following the new 'transformers'
        convention. In the old versions, this would instead be a standard collate_batch() function.
        """
        examples_ = examples.clone()
        batch = self._preprocess_batch(examples_)
        input_ids, labels = self.rotate_document(input_ids=batch)
        return input_ids, labels

    def rotate_document(self, input_ids):
        """
        Prepares rotated input and label pairs for document rotation pretext task.

        :param (torch.Tensor) input_ids: tensor with shape (B, S), contains sequences
        :return: (torch.Tensor, torch.Tensor) input_ids, labels
        """
        # In masked language modeling, the labels are the input itself (i.e. self-supervision)
        labels = input_ids.clone()

        # Store all of the special indices, those set by self.special_token_ids, in a mask
        all_special_indices = torch.zeros(labels.shape).bool()
        # Check if certain other token (e.g. [PAD], [UNK], etc.) exists in the tokenizer
        for special_token_id in self.special_token_ids:
            if special_token_id:
                # Get the related indices in the input IDs
                special_indices = labels.eq(special_token_id)
                all_special_indices |= special_indices

        # Everything except the special tokens should be visible in this task
        # TODO: Is this a correct approach? Can we do something smarter, and more efficient?
        #       Another approach would be to return the rotation index directly!
        labels[all_special_indices] = self.nomask_id

        # Do for each example in batch
        for i in range(labels.shape[0]):
            # Sample an index and rotate the entire example (i.e. document) around it
            rotation_index = torch.randint(low=0, high=labels.shape[1], size=(1,)).item()
            original_input_ids = input_ids.clone()  # NOTE: to prevent unsupported operation error
            input_ids[i, rotation_index:] = original_input_ids[i, :labels.shape[1] - rotation_index]
            input_ids[i, :rotation_index] = original_input_ids[i, labels.shape[1] - rotation_index:]

        # TODO: How can we circumvent this problem so we don't duplicate these calls?
        #       A good next step is to modularize the code for finding all special indices!
        #       It seems that we'll have to call it twice though, no way around?! -> THINK MORE!
        # NOTE: We do this again; input IDs are rotated and special tokens are in different places!
        # Store all of the special indices, those set by self.special_token_ids, in a mask
        all_special_indices = torch.zeros(labels.shape).bool()
        # Check if certain other token (e.g. [PAD], [UNK], etc.) exists in the tokenizer
        for special_token_id in self.special_token_ids:
            if special_token_id:
                # Get the related indices in the input IDs
                special_indices = input_ids.eq(special_token_id)
                all_special_indices |= special_indices

        # Do for each example in batch
        for i in range(labels.shape[0]):
            # Delete special tokens and add it to the end as [PAD]s; this is because [CLS] and [SEP]
            # tokens in the input will give the answer away in the case of document rotation.
            num_special_tokens = len(all_special_indices[i, :][all_special_indices[i, :]])
            input_ids[i, :] = torch.cat(
                [input_ids[i, :][~all_special_indices[i, :]],
                 torch.full(size=(num_special_tokens,), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)]
            )

        return input_ids, labels


class PermutationCollator(DataCollator):
    """
    Data collator used for permutation-based pretext tasks. It collates batches of tensors, honoring
    their tokenizer's pad_token and preprocesses batches by shuffling tokens or sets of tokens
    (e.g. sentences given by punctuation marks, etc.) randomly. The model learns to solve the puzzle
    to reconstruct the original image.

    :param (BertTokenizer) tokenizer: tokenizer for the logic expressions with encode() and decode()
    :param (int) nomask_id: integer IDs to indicate tokens that shouldn't be masked
    :param (None or list(int)) special_token_ids: list of integer IDs of tokens that are 'special';
            these tokens will not be masked and will not be replaced to ensure integrity of data.
            NOTE: Special tokens w.r.t. to the tokenizer are already taken care of thanks to the
            tokenizer.get_special_tokens_mask() function; this parameter gives a chance to extend it.
    """
    def __init__(self, tokenizer, permutation_policy='random', nomask_id=-100, special_token_ids=None):
        super(PermutationCollator).__init__()

        if tokenizer.mask_token is None:
            raise ValueError(' Tokenizer is missing the mask token (e.g. [MASK], <mask>, etc.)')

        self.tokenizer = tokenizer

        assert permutation_policy in ['random', 'sentence']
        self.permutation_policy = permutation_policy

        self.nomask_id = nomask_id

        self.special_token_ids = special_token_ids or [self.tokenizer.bos_token_id,
                                                       self.tokenizer.eos_token_id,
                                                       self.tokenizer.sep_token_id,
                                                       self.tokenizer.pad_token_id,
                                                       self.tokenizer.cls_token_id]

    def __call__(self, examples):
        """
        Makes the class callable, just like a function. This is following the new 'transformers'
        convention. In the old versions, this would instead be a standard collate_batch() function.
        """
        examples_ = examples.clone()
        batch = self._preprocess_batch(examples_)
        input_ids, labels = self.permute(input_ids=batch)
        return input_ids, labels

    def permute(self, input_ids):
        """
        Prepares permuted input and label pairs for permutation pretext task.

        :param (torch.Tensor) input_ids: tensor with shape (B, S), contains sequences
        :return: (torch.Tensor, torch.Tensor) input_ids, labels
        """
        # In masked language modeling, the labels are the input itself (i.e. self-supervision)
        labels = input_ids.clone()

        # Store all of the special indices, those set by self.special_token_ids, in a mask
        all_special_indices = torch.zeros(labels.shape).bool()
        # Check if certain other token (e.g. [PAD], [UNK], etc.) exists in the tokenizer
        for special_token_id in self.special_token_ids:
            if special_token_id:
                # Get the related indices in the input IDs
                special_indices = labels.eq(special_token_id)
                all_special_indices |= special_indices

        # Everything except the special tokens should be visible in this task
        # TODO: Is this a correct approach? Can we do something smarter, and more efficient?
        #       Another approach would be to return the rotation index directly!
        labels[all_special_indices] = self.nomask_id

        # Do for each example in batch
        for i in range(labels.shape[0]):
            if self.permutation_policy == 'random':
                permuted_indices = torch.randperm(n=labels.shape[1])
                input_ids[i, :] = input_ids[i, :][permuted_indices]
            elif self.permutation_policy == 'sentence':
                # TODO: Implement this permutation policy as well as potentially others
                pass

        # TODO: How can we circumvent this problem so we don't duplicate these calls?
        #       A good next step is to modularize the code for finding all special indices!
        #       It seems that we'll have to call it twice though, no way around?! -> THINK MORE!
        # NOTE: We do this again; input IDs are rotated and special tokens are in different places!
        # Store all of the special indices, those set by self.special_token_ids, in a mask
        all_special_indices = torch.zeros(labels.shape).bool()
        # Check if certain other token (e.g. [PAD], [UNK], etc.) exists in the tokenizer
        for special_token_id in self.special_token_ids:
            if special_token_id:
                # Get the related indices in the input IDs
                special_indices = input_ids.eq(special_token_id)
                all_special_indices |= special_indices

        # Do for each example in batch
        for i in range(labels.shape[0]):
            # Delete special tokens and add it to the end as [PAD]s; this is because [CLS] and [SEP]
            # tokens in the input will give the answer away in the case of document rotation.
            num_special_tokens = len(all_special_indices[i, :][all_special_indices[i, :]])
            input_ids[i, :] = torch.cat(
                [input_ids[i, :][~all_special_indices[i, :]],
                 torch.full(size=(num_special_tokens,), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)]
            )

        return input_ids, labels
