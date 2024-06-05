


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

path = '/Users/chaofeng/code/chat-t5/model_save/checkpoint-10'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSeq2SeqLM.from_pretrained(path)

txt = '你好吗？'
encode_txt = tokenizer(txt, return_tensors='pt')
'''
{'input_ids': tensor([[   6, 1683,  657,    6,   89]]), 
'token_type_ids': tensor([[0, 0, 0, 0, 0]]), 
'attention_mask': tensor([[1, 1, 1, 1, 1]])}
'''

input_ids = encode_txt.input_ids

outputs = model.generate(inputs=input_ids, max_new_tokens=40, do_sample=True)
res = tokenizer.decode_batch(sequences=outputs, )






