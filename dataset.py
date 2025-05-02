# Placeholder for dataset loading and preprocessing 

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging
import re
from tqdm import tqdm
import json
import platform

# 判断当前操作系统
IS_WINDOWS = platform.system().lower().startswith('win')
IS_LINUX = platform.system().lower() == 'linux'

# 读取配置文件，获取标注文件和图片目录的绝对路径
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def safe_path(path):
    """自动规范化路径，兼容Windows和Linux"""
    return os.path.normpath(path)

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError('未找到配置文件 config.json，请参考README.md在项目根目录下创建，并填写数据路径。')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

ANNO_DIR = safe_path(config.get('anno_dir', None))
IMG_DIR = safe_path(config.get('img_dir', None))

if not ANNO_DIR or not os.path.exists(ANNO_DIR):
    raise FileNotFoundError(f'标注文件目录不存在或未配置，请检查 config.json 中的 "anno_dir" 路径: {ANNO_DIR} (当前系统: {'Windows' if IS_WINDOWS else 'Linux'})')
if not IMG_DIR or not os.path.exists(IMG_DIR):
    raise FileNotFoundError(f'高分辨率图片目录不存在或未配置，请检查 config.json 中的 "img_dir" 路径: {IMG_DIR} (当前系统: {'Windows' if IS_WINDOWS else 'Linux'})')

# --- 配置日志记录 ---
# 1. 配置基本日志记录器 (INFO 及以上级别，输出到控制台)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 2. 配置专门用于记录验证警告的日志记录器 (输出到文件)
validation_log_file = 'validation_warnings.log'
validation_logger = logging.getLogger('validation_warnings')
validation_logger.setLevel(logging.WARNING) # 只记录 WARNING 及以上级别
# 防止重复添加 handler
if not validation_logger.handlers:
    val_file_handler = logging.FileHandler(validation_log_file, mode='w', encoding='utf-8')
    val_file_handler.setLevel(logging.WARNING)
    val_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    val_file_handler.setFormatter(val_formatter)
    validation_logger.addHandler(val_file_handler)
    validation_logger.propagate = False # 不将这些消息传递给根记录器 (控制台)

# 3. 新增：配置专门用于记录解析调试信息的日志记录器 (输出到文件)
debug_parsing_log_file = 'debug_parsing.log'
debug_parser_logger = logging.getLogger('debug_parsing')
debug_parser_logger.setLevel(logging.DEBUG) # 记录 DEBUG 及以上级别
# 防止重复添加 handler
if not debug_parser_logger.handlers:
    debug_file_handler = logging.FileHandler(debug_parsing_log_file, mode='w', encoding='utf-8')
    debug_file_handler.setLevel(logging.DEBUG)
    # 使用更简单的格式，因为我们主要关心消息本身
    debug_formatter = logging.Formatter('%(message)s') 
    debug_file_handler.setFormatter(debug_formatter)
    debug_parser_logger.addHandler(debug_file_handler)
    debug_parser_logger.propagate = False # 不将这些消息传递给根记录器 (控制台)

# --- 常量定义 ---
# 不再需要 DEFAULT_DATA_ROOT 作为核心逻辑的一部分
# DEFAULT_DATA_ROOT = r"E:\AIModels\DeepFashion\DeepFashion\Category and Attribute Prediction Benchmark"
ANNOTATION_DIR_NAME = "Anno_fine" # 这个可能仍然用于错误消息，暂时保留
IMAGE_DIR_NAME = os.path.join("Img", "img_highres") # 这个也可能用于错误消息

# 不再使用全局文件，优先使用分区文件
# PARTITION_FILE = "list_eval_partition.txt" # 已移除
# CATEGORY_IMG_FILE = "list_category_img.txt" # 可能冗余
# ATTRIBUTE_IMG_FILE = "list_attr_img.txt" # 可能冗余

CATEGORY_NAMES_FILE = "list_category_cloth.txt"
ATTRIBUTE_NAMES_FILE = "list_attr_cloth.txt"

# 定义分区文件名的模板
PARTITION_LIST_TEMPLATE = "{partition}.txt"
PARTITION_CATE_TEMPLATE = "{partition}_cate.txt"
PARTITION_ATTR_TEMPLATE = "{partition}_attr.txt"
# --- 常量定义结束 ---

class DeepFashionDataset(Dataset):
    """基于行对应假设加载 DeepFashion Category and Attribute Prediction Benchmark 数据集的 Dataset 类"""

    def __init__(self, anno_dir_path, image_dir_path, partition='train', transform=None):
        """
        Args:
            anno_dir_path (str): Anno_fine 目录的绝对路径。
            image_dir_path (str): 包含高分辨率图片的目录 (如 img_highres) 的绝对路径。
            partition (str): 加载的数据分区 ('train', 'val', 或 'test').
            transform (callable, optional): 应用于图像样本的可选变换.
        """
        if partition not in ['train', 'val', 'test']:
            raise ValueError(f"错误：无效的分区 '{partition}'. 必须是 'train', 'val', 或 'test'.")
        if not os.path.isdir(anno_dir_path):
            raise FileNotFoundError(f"错误：指定的 Anno_fine 目录未找到: {anno_dir_path}")
        if not os.path.isdir(image_dir_path):
            logging.warning(f"警告：指定的图片目录未找到或不是目录: {image_dir_path}")

        self.anno_dir = anno_dir_path
        self.image_dir = image_dir_path
        self.partition = partition
        self.transform = transform

        # --- 核心加载逻辑变更 --- 
        # 1. 按顺序加载图片标识符列表
        self.image_identifiers = self._load_image_list()
        if not self.image_identifiers:
             raise ValueError(f"错误：无法从 '{os.path.join(self.anno_dir, PARTITION_LIST_TEMPLATE.format(partition=self.partition))}' 加载 '{self.partition}' 分区的图片列表，或列表为空。")
        num_identifiers = len(self.image_identifiers)

        # 2. 加载类别和属性名称 (不变)
        self.category_names = self._load_names(CATEGORY_NAMES_FILE)
        self.attribute_names = self._load_names(ATTRIBUTE_NAMES_FILE)

        # 3. 按顺序加载类别 ID 列表
        category_ids = self._load_partition_category_list()
        if category_ids is None: # 函数内部出错会返回 None
             raise RuntimeError(f"错误：加载分区 '{self.partition}' 的类别 ID 列表失败。")
        num_categories = len(category_ids)
        
        # 4. 按顺序加载属性列表
        attribute_lists = self._load_partition_attribute_list()
        if attribute_lists is None:
            raise RuntimeError(f"错误：加载分区 '{self.partition}' 的属性列表失败。")
        num_attributes = len(attribute_lists)

        # 5. **关键校验：检查行数是否匹配**
        logging.info(f"分区 '{self.partition}': 图片标识符数量={num_identifiers}, 类别ID数量={num_categories}, 属性列表数量={num_attributes}")
        if not (num_identifiers == num_categories == num_attributes):
            error_msg = (
                f"错误：分区 '{self.partition}' 的文件行数不匹配! "
                f"标识符({PARTITION_LIST_TEMPLATE.format(partition=self.partition)}): {num_identifiers}, "
                f"类别({PARTITION_CATE_TEMPLATE.format(partition=self.partition)}): {num_categories}, "
                f"属性({PARTITION_ATTR_TEMPLATE.format(partition=self.partition)}): {num_attributes}. "
                f"这违反了按行严格对应的假设，请检查标注文件！"
            )
            logging.error(error_msg)
            raise ValueError(error_msg)
        else:
            logging.info(f"分区 '{self.partition}' 行数校验通过，共 {num_identifiers} 条有效数据。")

        # 6. **重建映射字典**
        self.category_map = {}
        self.attribute_map = {}
        try:
            logging.info(f"正在根据行对应关系重建 '{self.partition}' 分区的映射字典...")
            for i in range(num_identifiers):
                identifier = self.image_identifiers[i]
                self.category_map[identifier] = category_ids[i]
                self.attribute_map[identifier] = attribute_lists[i]
            logging.info(f"映射字典重建完成。Category Map 大小: {len(self.category_map)}, Attribute Map 大小: {len(self.attribute_map)}")
        except IndexError as e:
             # 理论上长度校验后不应发生，但作为保险
             logging.error(f"错误：重建映射时发生索引错误 (i={i})，这不应该发生！错误: {e}", exc_info=True)
             raise RuntimeError("重建映射时发生内部错误。")
        # --- 加载逻辑变更结束 ---

        logging.info(f"成功初始化 '{self.partition}' 分区 Dataset (基于行对应)。")

    def _load_image_list(self):
        """按顺序从分区特定的 .txt 文件加载图片标识符列表。"""
        list_filename = PARTITION_LIST_TEMPLATE.format(partition=self.partition)
        list_filepath = os.path.join(self.anno_dir, list_filename)
        identifiers = [] 
        if not os.path.exists(list_filepath):
            logging.error(f"错误：分区图像列表文件未找到: {list_filepath}")
            return identifiers
        try:
            with open(list_filepath, 'r', encoding='utf-8') as f:
                 # --- 文件头处理：移除！假设 list 文件没有头 --- 
                 # lines_to_process = []
                 # try:
                 #     first_line = next(f).strip()
                 #     # ... (之前的头处理逻辑已移除)
                 # except StopIteration:
                 #     logging.warning(f"警告：{list_filename} 文件为空。")
                 #     return identifiers
                 
                 # 直接读取所有行
                 lines_to_process = f.readlines()
                 # --- 文件头处理结束 ---
                 
                 # --- 调试：记录前 5 个有效数据行 --- 
                 debug_parser_logger.debug(f"--- {list_filename} - 文件前 5 个有效数据行原始内容 (读取后) ---")
                 count = 0
                 first_5_identifiers_from_list = []
                 for line in lines_to_process:
                     line = line.strip()
                     if line and not line.startswith('#'): # 仍然检查注释和空行
                         if count < 5:
                             debug_parser_logger.debug(f"  数据行 (原始): '{line}'")
                             first_5_identifiers_from_list.append(line)
                             count += 1
                 
                 if first_5_identifiers_from_list:
                    debug_parser_logger.debug(f"--- {list_filename} - 记录的前 5 个有效标识符 ---")
                    for ident in first_5_identifiers_from_list:
                        debug_parser_logger.debug(f"  - '{ident}'")
                 # --- 调试结束 ---
                 
                 # --- 正式加载所有标识符到列表 --- 
                 identifiers = [] # 重新初始化，确保为空
                 for line in lines_to_process:
                     line = line.strip()
                     if line and not line.startswith('#'):
                         identifiers.append(line) 

            logging.info(f"从 {list_filename} 加载了 {len(identifiers)} 个图片标识符 (保持顺序)。")
            return identifiers 
        except Exception as e:
            logging.error(f"错误：读取分区列表文件 {list_filepath} 时出错: {e}", exc_info=True)
            return None # 返回 None 表示失败

    def _load_names(self, filename):
        """加载类别或属性名称文件。(保持不变)"""
        filepath = os.path.join(self.anno_dir, filename)
        names = {}
        if not os.path.exists(filepath):
            logging.warning(f"警告：名称文件未找到: {filepath}")
            return names
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                next(f) # 跳过计数行
                next(f) # 跳过表头行
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 1:
                         names[i + 1] = parts[0] # ID 从 1 开始
                    else:
                         logging.warning(f"警告：跳过无效行（'{filename}' 第 {i+3} 行）: '{line.strip()}'")
            logging.info(f"从 {filename} 加载了 {len(names)} 个名称。")
            return names
        except Exception as e:
            logging.error(f"错误：读取名称文件 {filepath} 时出错: {e}", exc_info=True)
            return {} # 出错返回空字典

    def _load_partition_category_list(self):
        """按顺序从分区特定的 _cate.txt 文件加载类别 ID 列表 (假设无文件头)。"""
        cate_filename = PARTITION_CATE_TEMPLATE.format(partition=self.partition)
        cate_filepath = os.path.join(self.anno_dir, cate_filename)
        category_ids = [] 
        if not os.path.exists(cate_filepath):
            logging.error(f"错误：分区类别文件未找到: {cate_filepath}")
            return None 
        try:
            with open(cate_filepath, 'r', encoding='utf-8') as f:
                # --- 文件头处理：移除！假设无文件头 --- 
                # try:
                #     num_items_header = next(f).strip()
                #     column_header = next(f).strip()
                #     logging.info(f"从 {cate_filename} 跳过文件头: '{num_items_header}', '{column_header}'")
                # except StopIteration: ...
                lines_to_process = f.readlines() # 直接读取所有行
                # --- 文件头处理结束 ---

                # --- 调试：记录前 5 个有效数据行 (现在是 ID) ---
                debug_parser_logger.debug(f"--- {cate_filename} - 文件前 5 个有效数据行原始内容 (读取后, 假设无头) ---")
                count = 0
                for line in lines_to_process:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if count < 5:
                            debug_parser_logger.debug(f"  数据行 (原始ID): '{line}'")
                            count += 1
                # --- 调试结束 ---
                
                # --- 正式加载所有类别 ID --- 
                line_number = 0 # 从第一行开始计数
                for line in lines_to_process:
                    line_number += 1
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    try:
                        category_id = int(line)
                        category_ids.append(category_id)
                    except ValueError:
                        logging.warning(f"警告：无法将行内容解析为类别 ID（'{cate_filename}' 第 {line_number} 行）: '{line}'")
            logging.info(f"从 {cate_filename} 加载了 {len(category_ids)} 个类别 ID (保持顺序)。")
            return category_ids
        except Exception as e:
            logging.error(f"错误：读取分区类别文件 {cate_filepath} 时出错: {e}", exc_info=True)
            return None

    def _load_partition_attribute_list(self):
        """按顺序从分区特定的 _attr.txt 文件加载属性列表 (假设无文件头)。"""
        attr_filename = PARTITION_ATTR_TEMPLATE.format(partition=self.partition)
        attr_filepath = os.path.join(self.anno_dir, attr_filename)
        attribute_lists = [] 
        # --- 修改：预期属性数量改为 26 --- 
        num_expected_attributes = 26 
        # --- 修改结束 ---
        if not os.path.exists(attr_filepath):
            logging.error(f"错误：分区属性文件未找到: {attr_filepath}")
            return None 
        try:
            with open(attr_filepath, 'r', encoding='utf-8') as f:
                 # --- 文件头处理：移除！假设无文件头 --- 
                 # try:
                 #     num_items_header = next(f).strip()
                 #     column_header = next(f).strip()
                 #     logging.info(f"从 {attr_filename} 跳过文件头: '{num_items_header}', '{column_header}'")
                 # except StopIteration: ...
                 lines_to_process = f.readlines() # 直接读取所有行
                 
                 # --- 调试：记录前 5 个有效数据行 (现在是属性序列) ---
                 debug_parser_logger.debug(f"--- {attr_filename} - 文件前 5 个有效数据行原始内容 (读取后, 假设无头) ---")
                 count = 0
                 for line in lines_to_process:
                     line = line.strip()
                     if line and not line.startswith('#'):
                         if count < 5:
                             debug_parser_logger.debug(f"  数据行 (原始属性): '{line[:100]}...'") # 截断显示
                             count += 1
                 # --- 调试结束 ---
                     
                 # --- 正式加载所有属性列表 --- 
                 line_number = 0 
                 for line in lines_to_process:
                    line_number += 1
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    parts = line.split() 
                    try:
                        attributes = [int(attr) for attr in parts]
                        # --- 检查属性数量是否等于 26 --- 
                        if len(attributes) == num_expected_attributes:
                            attribute_lists.append(attributes)
                        else:
                            logging.warning(f"警告：解析得到的属性数量 ({len(attributes)}) 不等于预期的 {num_expected_attributes} 个（'{attr_filename}' 第 {line_number} 行）: '{line[:100]}...'")
                            continue 
                    except ValueError:
                        logging.warning(f"警告：无法解析部分或全部属性值为整数（'{attr_filename}' 第 {line_number} 行）: '{line[:100]}...'")
                        continue 
                    except Exception as inner_e:
                         logging.warning(f"警告：处理属性值时未知错误（'{attr_filename}' 第 {line_number} 行）: '{line[:100]}...', Error: {inner_e}")
                         continue 
            logging.info(f"从 {attr_filename} 加载了 {len(attribute_lists)} 个属性列表 (保持顺序)。")
            return attribute_lists
        except Exception as e:
            logging.error(f"错误：读取分区属性文件 {attr_filepath} 时出错: {e}", exc_info=True)
            return None
    
    def _get_image_path(self, identifier):
        """根据图片标识符构建实际的图片文件路径。(保持不变)"""
        if identifier.startswith("img/"):
             relative_path = identifier[4:]
        else:
             # 如果标识符格式变了，这里可能也需要调整，但暂时保留
             logging.warning(f"警告：图片标识符 '{identifier}' 不是以 'img/' 开头，将直接用于路径构建。")
             relative_path = identifier
        img_path = os.path.join(self.image_dir, relative_path.replace('/', os.sep))
        return img_path

    def __len__(self):
        """返回数据集中样本的数量 (基于标识符列表)"""
        return len(self.image_identifiers)

    def __getitem__(self, idx):
        """根据索引 idx 获取单个数据样本 (图像和标签)"""
        if idx < 0 or idx >= len(self.image_identifiers):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.image_identifiers) - 1}]")

        # --- 获取标识符、类别、属性的方式不变 (通过索引再查map) ---
        img_identifier = self.image_identifiers[idx]

        try:
            img_path = self._get_image_path(img_identifier)
        except Exception as e:
             logging.error(f"错误：为标识符 '{img_identifier}' 构建路径时出错: {e}")
             return {'error': f"无法构建路径: {e}", 'identifier': img_identifier}

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            logging.error(f"错误：图片文件未找到 (标识符: '{img_identifier}'): {img_path}")
            return {'error': f"图片未找到: {img_path}", 'identifier': img_identifier}
        except Exception as e:
            logging.error(f"错误：加载图片 {img_path} (标识符: '{img_identifier}') 时出错: {e}", exc_info=True)
            return {'error': f"加载图片时出错: {e}", 'identifier': img_identifier}

        # --- 使用重建后的 map 获取标签 --- 
        # 理论上 .get() 不应返回默认值，因为 __init__ 已校验并填充
        category_id = self.category_map.get(img_identifier, -999) # 用特殊值标记内部错误
        attributes = self.attribute_map.get(img_identifier) 

        # --- 错误检查：如果 get 返回了默认值，说明 map 构建逻辑有误 --- 
        if category_id == -999:
             internal_error_msg = f"内部错误：标识符 '{img_identifier}' 在 category_map 中未找到！这不应该在初始化校验后发生。"
             logging.error(internal_error_msg)
             validation_logger.error(internal_error_msg) # 也记录到验证日志
             category_id = -1 # 提供给下游的默认值
             # return {'error': internal_error_msg, 'identifier': img_identifier} # 可以选择直接报错返回
        if attributes is None:
             internal_error_msg = f"内部错误：标识符 '{img_identifier}' 在 attribute_map 中未找到！这不应该在初始化校验后发生。"
             logging.error(internal_error_msg)
             validation_logger.error(internal_error_msg)
             attributes = [-1] * 26 # 提供给下游的默认值
             # return {'error': internal_error_msg, 'identifier': img_identifier}

        # --- 转换为 Tensor (类别 ID 减 1 逻辑保持不变) --- 
        category_tensor = torch.tensor(category_id - 1 if category_id > 0 else -1, dtype=torch.long)
        attribute_tensor = torch.tensor(attributes, dtype=torch.float32)

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                 logging.error(f"错误：对图片 {img_path} (标识符: '{img_identifier}') 应用转换时出错: {e}", exc_info=True)
                 return {'error': f"转换图片时出错: {e}", 'identifier': img_identifier}

        sample = {
            'image': image,
            'category': category_tensor, 
            'attributes': attribute_tensor, 
            'identifier': img_identifier
        }

        return sample

# --- 示例用法 (保持不变，但期望不再报之前的格式错误) ---
if __name__ == '__main__':
    import sys
    output_log_file = 'dataset_run_output.log'
    print(f"测试 DeepFashionDataset (手动路径，基于行对应假设)，输出将被重定向到 {output_log_file} ...")

    test_anno_dir = r"E:\AIModels\DeepFashion\DeepFashion\Category and Attribute Prediction Benchmark\Anno_fine"
    test_image_dir = r"E:\AIModels\DeepFashion\DeepFashion\Category and Attribute Prediction Benchmark\Img\img_highres"

    original_stdout = sys.stdout
    try:
        with open(output_log_file, 'w', encoding='utf-8') as f_out:
            sys.stdout = f_out
            test_transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])
            partitions_to_test = ['train', 'val']
            for part in partitions_to_test:
                try:
                    print(f"\n--- 测试加载 {part} 分区 --- ")
                    print(f"  Anno Dir: {test_anno_dir}")
                    print(f"  Image Dir: {test_image_dir}")
                    dataset = DeepFashionDataset(
                        anno_dir_path=test_anno_dir, 
                        image_dir_path=test_image_dir, 
                        partition=part, 
                        transform=test_transform
                    )
                    print(f"{part} 分区样本数量: {len(dataset)}")
                    if len(dataset) > 0:
                        for index_to_test in [0, len(dataset) // 2, len(dataset) - 1]: # 测试开头、中间、结尾
                            print(f"  正在获取样本 {index_to_test}...")
                            sample = dataset[index_to_test]
                            if 'error' in sample:
                                 print(f"  获取样本 {index_to_test} 时出错: {sample['error']}")
                                 print(f"  标识符: {sample.get('identifier')}")
                            else:
                                print(f"  样本 {index_to_test} 信息:")
                                print(f"    图片 Tensor 尺寸: {sample['image'].shape}")
                                print(f"    类别 Tensor: {sample['category']} (ID: {sample['category'].item() + 1 if sample['category'].item() != -1 else 'N/A'})")
                                # 检查属性是否为默认值 [-1]*26
                                if sample['attributes'][0].item() == -1 and len(sample['attributes']) == 26 and torch.all(sample['attributes'] == -1):
                                     print(f"    属性 Tensor: (默认值 [-1] * 26)")
                                else:
                                     print(f"    属性 Tensor 形状: {sample['attributes'].shape}")
                                     print(f"    属性 Tensor 前10个: {sample['attributes'][:10].tolist()}")
                                print(f"    标识符: {sample['identifier']}")
                                if torch.max(sample['image']) > 1.0 or torch.min(sample['image']) < 0.0:
                                    print(f"    警告：图片 Tensor 的值超出 [0, 1] 范围")
                    else:
                        print(f"{part} 分区为空或加载失败，无法获取样本。")
                except (ValueError, RuntimeError) as ve:
                     print(f"加载 {part} 分区时发生配置或内部错误: {ve}")
                except FileNotFoundError as fnfe:
                     print(f"加载 {part} 分区时发生文件未找到错误: {fnfe}")
                except Exception as e:
                    print(f"加载或测试 {part} 分区时发生未知错误: {e}", exc_info=True)
            print("\n测试完成.")
    finally:
        sys.stdout = original_stdout
        print(f"测试输出已写入 {output_log_file}") 