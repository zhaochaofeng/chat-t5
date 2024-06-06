
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

path = '/Users/chaofeng/code/chat-t5/model_save/'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSeq2SeqLM.from_pretrained(path)

txt = '请介绍一下什么是医学'
inputs = tokenizer(text=txt, return_tensors='pt')
outputs = model.generate(inputs=inputs.input_ids, max_new_tokens=300, do_sample=True)
tokenizer.batch_decode(sequences=outputs, skip_special_tokens=True)




