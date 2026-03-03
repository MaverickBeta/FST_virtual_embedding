import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import warnings

# 忽略不必要的警告
warnings.filterwarnings("ignore")

model_id = "williamconvertino/qa_fst_1_3B"

print("⏳ 正在加载 Tokenizer...")
# 添加 trust_remote_code=True 自动信任自定义代码，免去输入 'y' 的麻烦
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("⏳ 正在修复配置并加载模型到 A6000...")
# 1. 提前加载 Config 配置文件
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
# 2. 手动打补丁：补全作者漏写的 tie_word_embeddings 属性，默认设为 False
config.tie_word_embeddings = False 

# 3. 带上修复后的 config 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    config=config,
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    dtype=torch.bfloat16,         # 传入 dtype 以消除作者代码中的 deprecated 警告
    trust_remote_code=True        # 自动信任自定义模型代码
)

print("✅ 模型加载成功！现在你可以开始提问了 (输入 'quit' 或 'exit' 退出)。\n")
print("-" * 50)

while True:
    user_input = input("你: ")
    if user_input.lower() in ['quit', 'exit']:
        print("退出对话。")
        break
    if not user_input.strip():
        continue

    prompt = f"Question: {user_input}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs, 
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print(f"QA Model: {response.strip()}\n")