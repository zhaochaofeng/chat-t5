
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from accelerate import Accelerator

# Initialize the accelerator
accelerator = Accelerator()

# Load dataset
dataset = load_dataset("imdb")

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
optimizer = AdamW(model.parameters(), lr=5e-5)

# Prepare everything with the accelerator
model, optimizer, dataset = accelerator.prepare(model, optimizer, dataset)

# Set up learning rate scheduler
num_training_steps = len(dataset['train']) // accelerator.gradient_accumulation_steps

lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Register components for checkpointing
accelerator.register_for_checkpointing(model)
accelerator.register_for_checkpointing(optimizer)
accelerator.register_for_checkpointing(lr_scheduler)

# Training loop
for epoch in range(1):
    for step, batch in enumerate(dataset['train']):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        if step % accelerator.gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    # Save checkpoint
    accelerator.save_state(f"checkpoint_epoch_{epoch}.bin")
print("Training completed.")

# Optionally load the checkpoint
accelerator.load_state(f"checkpoint_epoch_{epoch}.bin")
print('-' * 100)

