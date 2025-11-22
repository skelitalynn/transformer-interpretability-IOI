import openai
import json
import re
import time
from typing import List, Dict, Tuple
import random

class FreeFormIOIGenerator:
    """使用大模型自由生成IOI数据"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # 创建OpenAI客户端，但指向DeepSeek API端点
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"  # DeepSeek API端点
        )
    
    def generate_free_form_data(self, num_batches: int = 10, batch_size: int = 10) -> List[Dict]:
        """使用大模型自由生成IOI数据"""
        all_data = []
        
        for batch in range(num_batches):
            print(f"生成批次 {batch+1}/{num_batches}...")
            
            prompt = self._create_generation_prompt(batch_size)
            
            try:
                # 使用DeepSeek Reasoner模型
                response = self.client.chat.completions.create(
                    model="deepseek-chat",  # DeepSeek Reasoner模型
                    messages=[
                        {"role": "system", "content": "你是一个语言学实验数据生成专家。请严格按JSON格式输出。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=4000
                )
                
                content = response.choices[0].message.content
                batch_data = self._parse_response(content)
                all_data.extend(batch_data)
                
                # 避免速率限制
                time.sleep(1)
                
            except Exception as e:
                print(f"批次 {batch+1} 生成失败: {e}")
                continue
        
        return all_data
    
    def _create_generation_prompt(self, batch_size: int) -> str:
        """创建生成提示词"""
        return f"""
请生成 {batch_size} 个用于间接宾语识别(IOI)任务的句子对。

要求：
1. 每个句子对包含一个"正常"句子和一个"损坏"句子
2. 正常句子：描述某人将某物给某人或类似转移关系，句子以"to"结尾，下一个词应该是特定的人名或实体
3. 损坏句子：与正常句子相似，但通过改变某些关键信息，使得下一个词变成另一个不同的人名或实体
4. 句子要自然、多样化，包含不同的场景、动词和实体关系
5. 不要使用固定的模板，要创造性生成

示例：
{{
  "normal": "After the competition, the judge awarded the trophy to",
  "corrupted": "After the competition, the coach presented the medal to", 
  "normal_target": "the winner",
  "corrupted_target": "the captain"
}}

{{
  "normal": "When the meeting concluded, Sarah passed the notes to",
  "corrupted": "When the meeting concluded, David shared the document to",
  "normal_target": "her assistant",
  "corrupted_target": "the intern"
}}

{{
  "normal": "At the family gathering, grandmother gave the heirloom to",
  "corrupted": "At the family gathering, grandfather entrusted the watch to", 
  "normal_target": "her eldest granddaughter",
  "corrupted_target": "his grandson"
}}

请严格按照以下JSON格式输出，不要包含其他文字：
[
  {{
    "normal": "正常句子",
    "corrupted": "损坏句子",
    "normal_target": "预期下一个词",
    "corrupted_target": "损坏版本预期下一个词"
  }},
  ...
]
"""
    
    def _parse_response(self, content: str) -> List[Dict]:
        """解析模型响应"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                print("未找到JSON格式响应")
                return []
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"响应内容: {content}")
            return []
    
    def validate_and_clean(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """验证和清理数据"""
        valid_data = []
        invalid_data = []
        
        for item in data:
            try:
                normal = item["normal"].strip()
                corrupted = item["corrupted"].strip()
                normal_target = item["normal_target"].strip()
                corrupted_target = item["corrupted_target"].strip()
                
                # 基本验证
                if (len(normal) > 15 and len(corrupted) > 15 and
                    normal_target != corrupted_target and
                    len(normal_target) > 0 and len(corrupted_target) > 0 and
                    # 检查句子以"to"结尾（允许有些变体）
                    (normal.lower().endswith(' to') or normal.lower().endswith(' to"')) and
                    (corrupted.lower().endswith(' to') or corrupted.lower().endswith(' to"'))):
                    
                    # 清理句子结尾
                    if normal.endswith('"'):
                        normal = normal[:-1]
                    if corrupted.endswith('"'):
                        corrupted = corrupted[:-1]
                    
                    valid_data.append({
                        "normal": normal,
                        "corrupted": corrupted,
                        "normal_target": normal_target,
                        "corrupted_target": corrupted_target
                    })
                else:
                    invalid_data.append(item)
                    
            except (KeyError, AttributeError) as e:
                invalid_data.append(item)
        
        return valid_data, invalid_data

def generate_with_fallback():
    """如果API不可用，使用回退方法生成数据"""
    print("使用回退方法生成数据...")
    
    # 简单的回退数据
    fallback_data = [
        {
            "normal": "After the game, the coach gave the trophy to",
            "corrupted": "After the game, the captain passed the medal to", 
            "normal_target": "the MVP",
            "corrupted_target": "the rookie"
        },
        {
            "normal": "During the ceremony, the principal awarded the scholarship to",
            "corrupted": "During the ceremony, the donor presented the grant to",
            "normal_target": "the valedictorian", 
            "corrupted_target": "the athlete"
        },
        {
            "normal": "At the party, the host offered the gift to",
            "corrupted": "At the party, the guest brought the dessert to",
            "normal_target": "the guest of honor",
            "corrupted_target": "the host"
        },
        {
            "normal": "In the meeting, the manager delegated the task to",
            "corrupted": "In the meeting, the senior assigned the project to", 
            "normal_target": "the new intern",
            "corrupted_target": "the junior"
        },
        {
            "normal": "After the performance, the director gave the flowers to",
            "corrupted": "After the performance, the producer sent the contract to",
            "normal_target": "the lead actor",
            "corrupted_target": "the understudy"
        }
    ]
    
    return fallback_data

def main():
    API_KEY = "sk-4497179703b543aebf1a14ca19c340aa"
    
    if API_KEY == "your-openai-api-key-here":
        print("请设置有效的DeepSeek API密钥")
        dataset = generate_with_fallback()
    else:
        generator = FreeFormIOIGenerator(API_KEY)
        raw_data = generator.generate_free_form_data(num_batches=10, batch_size=8)
        dataset, invalid_data = generator.validate_and_clean(raw_data)
        
        if len(dataset) < 20:  # 如果有效数据太少，使用回退
            print("有效数据不足，使用回退数据...")
            dataset = generate_with_fallback()
    
    # 保存数据
    with open('free_form_ioi_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"生成完成！共 {len(dataset)} 条有效数据")
    print("\n样本数据:")
    for i, item in enumerate(dataset[:5]):
        print(f"\n{i+1}.")
        print(f"正常: {item['normal']}")
        print(f"损坏: {item['corrupted']}") 
        print(f"目标: '{item['normal_target']}' -> '{item['corrupted_target']}'")
    
    # 分析数据多样性
    print(f"\n数据统计:")
    target_types = {}
    for item in dataset:
        target = item["normal_target"]
        if target not in target_types:
            target_types[target] = 0
        target_types[target] += 1
    
    print(f"唯一目标词数量: {len(target_types)}")
    print(f"目标词示例: {list(target_types.keys())[:10]}")

if __name__ == "__main__":
    main()