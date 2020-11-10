from collators import MLMCollator, SequentialMLMCollator, PermutationMLMCollator, \
                      TextInfillingCollator, TokenDeletionCollator, DocumentRotationCollator, \
                      PermutationCollator

from transformers import BertTokenizer

from functools import partial

from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True,
                                          do_basic_tokenize=True,
                                          unk_token='[UNK]',
                                          sep_token='[SEP]',
                                          pad_token='[PAD]',
                                          cls_token='[CLS]',
                                          mask_token='[MASK]')

encode = partial(TOKENIZER.encode, max_length=48)
pad_token_id = TOKENIZER.convert_tokens_to_ids(TOKENIZER.pad_token)
# NOTE: We add the pad token ID as discussed in https://github.com/pytorch/text/issues/609
TEXT = Field(use_vocab=False, tokenize=encode, pad_token=pad_token_id, batch_first=True)
LABEL = LabelField()

train_dataset, test_dataset = IMDB.splits(text_field=TEXT, label_field=LABEL)
LABEL.build_vocab(train_dataset)
train_loader, test_loader = BucketIterator.splits(datasets=(train_dataset, test_dataset), batch_size=1)

# NOTE: We could have also passed the data collators through the collate_fn() function of the
#       data loader in normal circumstances. Here, we pass it inside the training loop
#       to adhere to the code of torchtext
collator1 = MLMCollator(tokenizer=TOKENIZER, mask_probability=0.25)
collator2 = SequentialMLMCollator(tokenizer=TOKENIZER)
collator3 = PermutationMLMCollator(tokenizer=TOKENIZER, mask_probability=0.15, max_span_length=5)
collator4 = TextInfillingCollator(tokenizer=TOKENIZER, mask_probability=0.15, mean_span_length=3, mask_strategy='spanbert')
collator5 = TextInfillingCollator(tokenizer=TOKENIZER, mask_probability=0.15, mean_span_length=3, mask_strategy='bart')
collator6 = TokenDeletionCollator(tokenizer=TOKENIZER, mask_probability=0.15)
collator7 = DocumentRotationCollator(tokenizer=TOKENIZER)
collator8 = PermutationCollator(tokenizer=TOKENIZER, permutation_policy='random')


for batch in train_loader:
    x, y = batch.text, batch.label

    """
    print('----------------------------MASKED LANGUAGE MODELING (BERT)----------------------------')
    examples, labels = collator1(x)
    print('%s\n->\n%s' % (TOKENIZER.decode(examples[0].tolist()), TOKENIZER.decode(labels[0].tolist())))

    print('--------------------------SEQUENTIAL MASKED LANGUAGE MODELING--------------------------')
    examples, labels = collator2(x)
    print('%s\n->\n%s' % (TOKENIZER.decode(examples[0].tolist()), TOKENIZER.decode(labels[0].tolist())))
    print('%s\n->\n%s' % (TOKENIZER.decode(examples[1].tolist()), TOKENIZER.decode(labels[1].tolist())))

    print('------------------PERMUTATION-BASED MASKED LANGUAGE MODELING (XLNET)-------------------')
    examples, permutation_mask, labels = collator3(x)
    print('%s\n->\n%s' % (TOKENIZER.decode(examples[0].tolist()), TOKENIZER.decode(labels[0].tolist())))
    print('Sample Permutation Mask Shape: ',  permutation_mask.shape)
   
    print('-------------------TEXT INFILLING MASKED LANGUAGE MODELING (SPANBERT)------------------')
    examples, labels = collator4(x)
    print('%s\n->\n%s' % (TOKENIZER.decode(examples[0].tolist()), TOKENIZER.decode(labels[0].tolist())))

    print('---------------------TEXT INFILLING MASKED LANGUAGE MODELING (BART)--------------------')
    examples, labels = collator5(x)
    print('%s\n->\n%s' % (TOKENIZER.decode(examples[0].tolist()), TOKENIZER.decode(labels[0].tolist())))
    
    print('---------------------------------TOKEN DELETION (BART)---------------------------------')
    examples, labels = collator6(x)
    print('%s\n->\n%s' % (TOKENIZER.decode(examples[0].tolist()), TOKENIZER.decode(labels[0].tolist())))
    
    print('--------------------------------DOCUMENT ROTATION (BART)-------------------------------')
    examples, labels = collator7(x)
    print('%s\n->\n%s' % (TOKENIZER.decode(examples[0].tolist()), TOKENIZER.decode(labels[0].tolist())))
    """

    print('-----------------------------------PERMUTATION (BART)----------------------------------')
    examples, labels = collator8(x)
    print('%s\n->\n%s' % (TOKENIZER.decode(examples[0].tolist()), TOKENIZER.decode(labels[0].tolist())))

    break


# TODO: Make sure that permutation MLM is actually useful besides XLNet!
# TODO: Make sure that text-infilling for 'bart' makes sense and is learnable!
# TODO: Make sure that token deletion makes sense and is learnable!
# TODO: FairSeq seems to have the implementations: https://github.com/pytorch/fairseq/blob/aa79bb9c37b27e3f84e7a4e182175d3b50a79041/fairseq/data/denoising_dataset.py
# TODO: Implement sentence permutation!
# TODO: These might be too hard tasks to solve as they stand!
#       (to-do lines indicate potential places of change in `collators.py`)
#       The question is how can we relax the assumptions?
