
'''
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

path = '/Users/chaofeng/code/chat-t5/model_save/sft/checkpoint-8000'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSeq2SeqLM.from_pretrained(path)

txt = '请介绍一下什么是机器学习'
inputs = tokenizer(text=txt, return_tensors='pt')
outputs = model.generate(inputs=inputs.input_ids, max_new_tokens=300, do_sample=True)
tokenizer.batch_decode(sequences=outputs, skip_special_tokens=True)
'''
'''
from rich.console import Console
from rich.text import Text

console = Console()
text = Text('Hello, World!')
text.stylize('bold magenta', start=0, end=6)
console.print(text)

# append函数
text = Text()
text.append('Hello', style='bold magenta')
text.append(' World!')
console.print(text)

# 解析ANSI格式数据
text = Text.from_ansi("\033[1mHello, World!\033[0m")
console.print(text.spans)

text = Text.assemble(('Hello', 'bold magenta'), ' World')
console.print(text)


from rich import print
from rich.panel import Panel
from rich.text import Text

panel = Panel(Text('Hello', justify='right'))
print(panel)
'''

import torch

model = torch.nn.Transformer()

