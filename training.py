import torch
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

# Load the SQuAD dataset from Hugging Face
dataset = load_dataset('squad')



def sample_dataset(dataset, fraction=1):
    return dataset.shuffle(seed=42).select([i for i in range(len(dataset)) if i % int(1 / fraction) == 0])



train_dataset = sample_dataset(dataset['train'])
valid_dataset = sample_dataset(dataset['validation'])


model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(
        examples['question'],
        examples['context'],
        truncation=True,
        padding='max_length',
        max_length=384,
        return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i, offset_mapping in enumerate(tokenized_inputs['offset_mapping']):
        
        answers = examples['answers']
        start_char = answers[i]['answer_start'][0]
        answer_text = answers[i]['text'][0]
        end_char = start_char + len(answer_text)

        start_token = None
        end_token = None

        for j, (start, end) in enumerate(offset_mapping):
            if start <= start_char < end:
                start_token = j
            if start < end_char <= end:
                end_token = j
            if start_token is not None and end_token is not None:
                break

        
        if start_token is None:
            start_token = len(offset_mapping) - 1
        if end_token is None:
            end_token = len(offset_mapping) - 1

        start_positions.append(start_token)
        end_positions.append(end_token)

    
    tokenized_inputs.update({
        'start_positions': start_positions,
        'end_positions': end_positions
    })
    return tokenized_inputs



def process_dataset(dataset):
    return dataset.map(tokenize_and_align_labels, batched=True)


tokenized_train_datasets = process_dataset(train_dataset)
tokenized_valid_datasets = process_dataset(valid_dataset)


tokenized_train_datasets.set_format('torch',
                                    columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
tokenized_valid_datasets.set_format('torch',
                                    columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])


training_args = TrainingArguments(
    output_dir='./results4',
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    logging_dir='./logs3',
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=50,
    fp16=True,  # Allow mixed precision training
    report_to='none',  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_valid_datasets
)

trainer.train()

results = trainer.evaluate()
print(results)

model.save_pretrained('finetuned_distilbert_squad_full')
tokenizer.save_pretrained('finetuned_distilbert_squad_full')
