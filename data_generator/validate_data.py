import json
import re
from typing import List, Dict, Tuple

class IOIDataValidator:
    """IOI数据格式验证器"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self):
        """初始化验证模式"""
        return {
            # 基本句子模式：以"to"结尾的完整句子
            "sentence_pattern": re.compile(r'^[A-Z][^.!?]*\bto$', re.IGNORECASE),
            
            # 目标词模式：可以是人名、角色名、关系词等
            "target_pattern": re.compile(r'^[a-zA-Z\s\'-]+$'),
        }
    
    def validate_single_item(self, item: Dict) -> Tuple[bool, List[str]]:
        """验证单个数据项"""
        errors = []
        
        # 检查必需字段是否存在
        required_fields = ['normal', 'corrupted', 'normal_target', 'corrupted_target']
        for field in required_fields:
            if field not in item:
                errors.append(f"缺少必需字段: {field}")
                return False, errors
        
        normal = item['normal'].strip()
        corrupted = item['corrupted'].strip()
        normal_target = item['normal_target'].strip()
        corrupted_target = item['corrupted_target'].strip()
        
        # 1. 检查句子格式
        if not self._validate_sentence_format(normal):
            errors.append(f"正常句子格式不正确: {normal}")
        if not self._validate_sentence_format(corrupted):
            errors.append(f"损坏句子格式不正确: {corrupted}")
        
        # 2. 检查目标词格式
        if not self._validate_target_format(normal_target):
            errors.append(f"正常目标词格式不正确: {normal_target}")
        if not self._validate_target_format(corrupted_target):
            errors.append(f"损坏目标词格式不正确: {corrupted_target}")
        
        # 3. 检查目标词是否不同
        if normal_target.lower() == corrupted_target.lower():
            errors.append(f"目标词相同: {normal_target} == {corrupted_target}")
        
        # 4. 检查句子长度（放宽限制）
        if len(normal.split()) < 4:
            errors.append(f"正常句子过短: {normal}")
        if len(corrupted.split()) < 4:
            errors.append(f"损坏句子过短: {corrupted}")
        
        # 5. 检查句子是否以"to"结尾（主要检查）
        if not normal.lower().rstrip('"').endswith(' to'):
            errors.append(f"正常句子不以'to'结尾: {normal}")
        if not corrupted.lower().rstrip('"').endswith(' to'):
            errors.append(f"损坏句子不以'to'结尾: {corrupted}")
        
        return len(errors) == 0, errors
    
    def _validate_sentence_format(self, sentence: str) -> bool:
        """验证句子格式"""
        # 清理句子结尾的引号
        sentence = sentence.rstrip('"')
        
        # 检查是否以大写字母开头，以"to"结尾
        if not sentence.strip().endswith(' to'):
            return False
        
        # 检查句子长度
        words = sentence.split()
        if len(words) < 4:  # 放宽到至少4个词
            return False
        
        return True
    
    def _validate_target_format(self, target: str) -> bool:
        """验证目标词格式"""
        # 目标词应该只包含字母、空格、连字符和撇号
        if not self.patterns["target_pattern"].match(target):
            return False
        
        # 目标词不应该为空
        if len(target.strip()) == 0:
            return False
        
        # 目标词长度应该合理（放宽限制）
        if len(target) > 50:  # 放宽到50个字符
            return False
        
        return True
    
    def validate_dataset(self, dataset: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """验证整个数据集"""
        valid_data = []
        invalid_data = []
        
        print("开始数据格式验证...")
        
        for i, item in enumerate(dataset):
            is_valid, errors = self.validate_single_item(item)
            
            if is_valid:
                valid_data.append(item)
            else:
                invalid_item = item.copy()
                invalid_item["validation_errors"] = errors
                invalid_data.append(invalid_item)
                
                if len(invalid_data) <= 5:  # 只显示前5个错误样本
                    print(f"无效样本 {i+1}: {errors}")
        
        return valid_data, invalid_data
    
    def generate_validation_report(self, valid_data: List[Dict], invalid_data: List[Dict]) -> Dict:
        """生成验证报告"""
        total_samples = len(valid_data) + len(invalid_data)
        
        report = {
            "total_samples": total_samples,
            "valid_samples": len(valid_data),
            "invalid_samples": len(invalid_data),
            "valid_ratio": len(valid_data) / total_samples if total_samples > 0 else 0,
            "error_breakdown": {},
            "sample_analysis": self._analyze_samples(valid_data)
        }
        
        # 统计错误类型
        error_counts = {}
        for item in invalid_data:
            for error in item.get("validation_errors", []):
                error_type = error.split(":")[0] if ":" in error else error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        report["error_breakdown"] = error_counts
        
        return report

    def _analyze_samples(self, valid_data: List[Dict]) -> Dict:
        """分析有效样本的特征"""
        if not valid_data:
            return {}
        
        analysis = {
            "avg_sentence_length": 0,
            "target_variety": set(),
            "sentence_patterns": set()
        }
        
        total_normal_length = 0
        total_corrupted_length = 0
        
        for item in valid_data:
            # 计算句子长度
            normal_words = len(item['normal'].split())
            corrupted_words = len(item['corrupted'].split())
            total_normal_length += normal_words
            total_corrupted_length += corrupted_words
            
            # 收集目标词
            analysis["target_variety"].add(item['normal_target'])
            analysis["target_variety"].add(item['corrupted_target'])
        
        analysis["avg_sentence_length"] = {
            "normal": total_normal_length / len(valid_data),
            "corrupted": total_corrupted_length / len(valid_data)
        }
        analysis["target_variety"] = list(analysis["target_variety"])
        
        return analysis

def main():
    # 加载生成的数据
    try:
        with open('free_form_ioi_dataset.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("数据文件不存在，请先运行数据生成脚本")
        return
    
    print(f"加载了 {len(dataset)} 条数据")
    
    # 验证数据
    validator = IOIDataValidator()
    valid_data, invalid_data = validator.validate_dataset(dataset)
    
    # 生成报告
    report = validator.generate_validation_report(valid_data, invalid_data)
    
    # 打印报告
    print("\n" + "="*50)
    print("数据验证报告")
    print("="*50)
    print(f"总样本数: {report['total_samples']}")
    print(f"有效样本: {report['valid_samples']}")
    print(f"无效样本: {report['invalid_samples']}")
    print(f"有效率: {report['valid_ratio']:.2%}")
    
    if report['error_breakdown']:
        print(f"\n错误分布:")
        for error_type, count in report['error_breakdown'].items():
            print(f"  {error_type}: {count} 次")
    
    if report['sample_analysis']:
        analysis = report['sample_analysis']
        print(f"\n样本分析:")
        print(f"  平均句子长度 - 正常: {analysis['avg_sentence_length']['normal']:.1f} 词")
        print(f"  平均句子长度 - 损坏: {analysis['avg_sentence_length']['corrupted']:.1f} 词")
        print(f"  唯一目标词数量: {len(analysis['target_variety'])}")
    
    # 保存验证后的数据
    if valid_data:
        with open('validated_ioi_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(valid_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 有效数据已保存到: validated_ioi_dataset.json")
    
    if invalid_data:
        with open('invalid_ioi_data.json', 'w', encoding='utf-8') as f:
            json.dump(invalid_data, f, indent=2, ensure_ascii=False)
        print(f"❌ 无效数据已保存到: invalid_ioi_data.json")
    
    # 显示一些有效样本
    print(f"\n有效样本示例:")
    for i, item in enumerate(valid_data[:3]):
        print(f"\n{i+1}.")
        print(f"  正常: {item['normal']}")
        print(f"  损坏: {item['corrupted']}")
        print(f"  目标: '{item['normal_target']}' -> '{item['corrupted_target']}'")

if __name__ == "__main__":
    main()