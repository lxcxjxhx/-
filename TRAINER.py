import os
import gc
import tkinter as tk
from tkinter import ttk, Toplevel, Text, Scrollbar, messagebox, filedialog
from ttkthemes import ThemedTk
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer, TrainerCallback, DataCollatorForLanguageModeling
from safetensors.torch import save_file
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset
import pandas as pd
import json
from threading import Thread
import psutil
import pynvml
from PIL import Image
import logging
import traceback
from datetime import datetime
import importlib.metadata  # 替换 pkg_resources
import sys

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# Set up logging
logging.basicConfig(
    filename="training.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Check library versions and availability
def check_library_versions():
    libraries = {
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
    }
    for lib, version in libraries.items():
        logging.info(f"{lib} version: {version}")
    return libraries

# Check if bitsandbytes is available for 4-bit quantization
try:
    import bitsandbytes
    BITSANDBYTES_AVAILABLE = True
    logging.info("bitsandbytes is available for 4-bit quantization")
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logging.warning("bitsandbytes is not available. 4-bit quantization will be disabled.")

# Check if optimum is available for model optimization
try:
    import optimum
    OPTIMUM_AVAILABLE = True
    logging.info("optimum is available for model optimization")
except ImportError:
    OPTIMUM_AVAILABLE = False
    logging.warning("optimum is not available. Some optimizations will be disabled.")

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=32):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.device = torch.device("cuda")  # 强制使用CUDA
        logging.info(f"初始化数据集: {data_path}")
        self.load_data()

    def load_data(self):
        try:
            if os.path.isdir(self.data_path):
                self._load_directory_data()
            else:
                self._load_file_data()
            
            if not self.data:
                raise ValueError("No valid data loaded from the file")
                
            logging.info(f"Successfully loaded {len(self.data)} records")
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}\n{traceback.format_exc()}")
            raise

    def _load_directory_data(self):
        """加载目录格式的数据集"""
        text_dir = os.path.join(self.data_path, "text")
        if not os.path.exists(text_dir):
            raise ValueError(f"Text directory not found: {text_dir}")
            
        for filename in os.listdir(text_dir):
            if filename.endswith(('.txt', '.md')):
                file_path = os.path.join(text_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    if text:
                        self.data.append({
                            'text': text,
                            'image': None
                        })
                except Exception as e:
                    logging.warning(f"Error reading file {filename}: {str(e)}")

    def _load_file_data(self):
        """加载文件格式的数据集"""
        file_ext = os.path.splitext(self.data_path)[1].lower()
        
        try:
            if file_ext == '.jsonl':
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            if isinstance(item, dict) and 'text' in item and item['text']:
                                self.data.append(item)
                        except json.JSONDecodeError as e:
                            logging.warning(f"Invalid JSON line: {str(e)}")
                            continue
                            
            elif file_ext == '.json':
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'text' in item and item['text']:
                                self.data.append(item)
                    elif isinstance(data, dict) and 'text' in data and data['text']:
                        self.data.append(data)
                        
            elif file_ext == '.csv':
                df = pd.read_csv(self.data_path)
                if 'text' in df.columns:
                    for _, row in df.iterrows():
                        if pd.notna(row['text']) and str(row['text']).strip():
                            self.data.append({
                                'text': str(row['text']).strip(),
                                'image': row.get('image') if 'image' in df.columns else None
                            })
                            
        except Exception as e:
            logging.error(f"Error reading file {self.data_path}: {str(e)}")
            raise

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            text = item['text']
            if not text or not isinstance(text, str):
                logging.warning(f"Encountered invalid text at index {idx}, using default text")
                text = "[UNK]"
                
            # 使用tokenizer处理文本
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 将所有张量移动到CUDA设备
            input_ids = encoding['input_ids'].squeeze().cuda()
            attention_mask = encoding['attention_mask'].squeeze().cuda()
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            output = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            
            # 处理图像数据
            if item.get('image'):
                try:
                    image = Image.open(item['image']).convert('RGB')
                    # 将图像转换为张量并移动到CUDA设备
                    image_tensor = torch.tensor(image).cuda()
                    output['pixel_values'] = image_tensor
                except Exception as e:
                    logging.warning(f"Error loading image at index {idx}: {str(e)}")
                    output['pixel_values'] = None
                    
            return output
            
        except Exception as e:
            logging.error(f"Error in __getitem__ at index {idx}: {str(e)}\n{traceback.format_exc()}")
            raise

    def __len__(self):
        return len(self.data)

# Progress Callback for Training
class ProgressCallback(TrainerCallback):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.total_steps = 0
        self.current_step = 0

    def on_init_end(self, args, state, control, **kwargs):
        logging.debug("Trainer initialization completed")
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps
        self.current_step = 0
        logging.debug(f"Training started with {self.total_steps} total steps")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self.current_step = state.global_step
        progress = (self.current_step / self.total_steps) * 100
        self.app.root.after(0, lambda: self.app.progress.config(value=progress))
        self.app.root.after(0, lambda: self.app.status.config(text=f"Training in progress... Step {self.current_step}/{self.total_steps}"))
        logging.debug(f"Step {self.current_step}/{self.total_steps} completed")
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        self.app.root.after(0, lambda: self.app.status.config(text=f"Epoch {state.epoch} completed"))
        logging.debug(f"Epoch {state.epoch} completed")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        logging.debug("Training ended")
        return control

# Custom Trainer to Debug Loss
class CustomTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # Debug loss
        logging.debug(f"Loss: {loss.item()}, requires_grad: {loss.requires_grad}, grad_fn: {loss.grad_fn}")

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

# Main Application Class
class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Transformer Training Studio")
        self.root.geometry("1000x800")  # 增加窗口大小
        self.root.resizable(True, True)

        # Check library versions
        check_library_versions()

        # Initialize GPU device count
        self.device_count = 0
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            logging.debug(f"检测到 {self.device_count} 个GPU设备")
        except Exception as e:
            logging.warning(f"初始化NVML失败: {str(e)}\n{traceback.format_exc()}")
            self.device_count = 0

        # Directories
        self.data_dir = "data"
        self.models_dir = "models"
        self.trained_models_dir = "trained_models"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.trained_models_dir, exist_ok=True)

        # Tkinter variables
        self.data_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.device_var = tk.StringVar(value="CPU")
        self.batch_size_var = tk.StringVar(value="1")
        self.epochs_var = tk.StringVar(value="1")
        self.learning_rate_var = tk.StringVar(value="2e-5")
        self.theme_var = tk.StringVar(value="arc")
        self.max_length_var = tk.StringVar(value="32")
        self.warmup_steps_var = tk.StringVar(value="0")
        self.weight_decay_var = tk.StringVar(value="0.01")
        self.gradient_accumulation_steps_var = tk.StringVar(value="1")
        self.training_thread = None
        self.trainer = None
        self.is_training = False
        self.is_paused = False

        # 存储Entry控件的引用
        self.entries = {}
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 标题
        title_label = ttk.Label(main_frame, text="Transformer Training Studio", font=("Helvetica", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # 创建左右分栏
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=5)
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=5)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # 左侧：数据集配置
        dataset_frame = ttk.LabelFrame(left_frame, text="数据集配置", padding="10")
        dataset_frame.grid(row=0, column=0, sticky="nsew", pady=5)
        left_frame.rowconfigure(0, weight=1)

        ttk.Label(dataset_frame, text="数据集:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.data_combo = ttk.Combobox(dataset_frame, textvariable=self.data_var, width=30)
        self.data_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.data_combo.bind("<<ComboboxSelected>>", lambda _: self.estimate_memory())
        ttk.Button(dataset_frame, text="浏览", command=self.browse_dataset).grid(row=0, column=2, padx=5)
        ttk.Button(dataset_frame, text="预览", command=self.preview_dataset).grid(row=0, column=3, padx=5)

        # 数据集格式转换
        format_frame = ttk.LabelFrame(dataset_frame, text="格式转换", padding="5")
        format_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=5)
        
        ttk.Label(format_frame, text="目标格式:").grid(row=0, column=0, sticky="e", padx=5)
        self.target_format_var = tk.StringVar(value="jsonl")
        format_combo = ttk.Combobox(format_frame, textvariable=self.target_format_var, values=["jsonl", "json", "csv"], width=10)
        format_combo.grid(row=0, column=1, sticky="w", padx=5)
        ttk.Button(format_frame, text="转换", command=self.convert_dataset).grid(row=0, column=2, padx=5)

        # 右侧：模型配置
        model_frame = ttk.LabelFrame(right_frame, text="模型配置", padding="10")
        model_frame.grid(row=0, column=0, sticky="nsew", pady=5)
        right_frame.rowconfigure(0, weight=1)

        ttk.Label(model_frame, text="模型:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, width=30)
        self.model_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.model_combo.bind("<<ComboboxSelected>>", lambda _: self.estimate_memory())
        ttk.Button(model_frame, text="浏览", command=self.browse_model).grid(row=0, column=2, padx=5)

        # 训练参数配置
        training_frame = ttk.LabelFrame(right_frame, text="训练参数", padding="10")
        training_frame.grid(row=1, column=0, sticky="nsew", pady=5)

        # 第一行参数
        ttk.Label(training_frame, text="设备:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        device_combo = ttk.Combobox(training_frame, textvariable=self.device_var, values=["CPU", "GPU"], width=10)
        device_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        device_combo.state(["readonly"])

        ttk.Label(training_frame, text="批次大小:").grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.entries['batch_size'] = ttk.Entry(training_frame, textvariable=self.batch_size_var, width=10)
        self.entries['batch_size'].grid(row=0, column=3, sticky="w", padx=5, pady=5)

        # 第二行参数
        ttk.Label(training_frame, text="轮数:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.entries['epochs'] = ttk.Entry(training_frame, textvariable=self.epochs_var, width=10)
        self.entries['epochs'].grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(training_frame, text="学习率:").grid(row=1, column=2, sticky="e", padx=5, pady=5)
        self.entries['learning_rate'] = ttk.Entry(training_frame, textvariable=self.learning_rate_var, width=10)
        self.entries['learning_rate'].grid(row=1, column=3, sticky="w", padx=5, pady=5)

        # 第三行参数
        ttk.Label(training_frame, text="最大长度:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.entries['max_length'] = ttk.Entry(training_frame, textvariable=self.max_length_var, width=10)
        self.entries['max_length'].grid(row=2, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(training_frame, text="预热步数:").grid(row=2, column=2, sticky="e", padx=5, pady=5)
        self.entries['warmup_steps'] = ttk.Entry(training_frame, textvariable=self.warmup_steps_var, width=10)
        self.entries['warmup_steps'].grid(row=2, column=3, sticky="w", padx=5, pady=5)

        # 第四行参数
        ttk.Label(training_frame, text="权重衰减:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.entries['weight_decay'] = ttk.Entry(training_frame, textvariable=self.weight_decay_var, width=10)
        self.entries['weight_decay'].grid(row=3, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(training_frame, text="梯度累积:").grid(row=3, column=2, sticky="e", padx=5, pady=5)
        self.entries['gradient_accumulation_steps'] = ttk.Entry(training_frame, textvariable=self.gradient_accumulation_steps_var, width=10)
        self.entries['gradient_accumulation_steps'].grid(row=3, column=3, sticky="w", padx=5, pady=5)

        # 资源使用情况
        resource_frame = ttk.LabelFrame(main_frame, text="资源使用", padding="10")
        resource_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)

        ttk.Label(resource_frame, text="预计内存:").grid(row=0, column=0, sticky="e", padx=5)
        self.ram_label = ttk.Label(resource_frame, text="0 MB")
        self.ram_label.grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(resource_frame, text="预计显存:").grid(row=1, column=0, sticky="e", padx=5)
        self.vram_label = ttk.Label(resource_frame, text="0 MB")
        self.vram_label.grid(row=1, column=1, sticky="w", padx=5)

        # 训练进度
        progress_frame = ttk.LabelFrame(main_frame, text="训练进度", padding="10")
        progress_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)

        self.progress = ttk.Progressbar(progress_frame, length=800, mode='determinate', maximum=100)
        self.progress.grid(row=0, column=0, columnspan=3, pady=5)

        self.status = ttk.Label(progress_frame, text="就绪", font=("Helvetica", 10, "italic"))
        self.status.grid(row=1, column=0, columnspan=3, pady=5)

        # 控制按钮
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=4, column=0, columnspan=2, pady=10)

        self.start_button = ttk.Button(control_frame, text="开始训练", command=self.start_training, style="Accent.TButton")
        self.start_button.grid(row=0, column=0, padx=5)

        self.pause_button = ttk.Button(control_frame, text="暂停", command=self.pause_training, state="disabled")
        self.pause_button.grid(row=0, column=1, padx=5)

        self.cancel_button = ttk.Button(control_frame, text="取消", command=self.cancel_training, state="disabled")
        self.cancel_button.grid(row=0, column=2, padx=5)

        # 创建菜单栏
        self.create_menu()

        # 创建工具提示
        self.create_tooltips()

        # 设置样式
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Helvetica", 10, "bold"))

        self.update_data_list()
        self.update_model_list()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="保存配置", command=self.save_config)
        file_menu.add_command(label="加载配置", command=self.load_config)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing)

        # 视图菜单
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)
        view_menu.add_command(label="查看日志", command=self.view_logs)
        view_menu.add_separator()
        view_menu.add_radiobutton(label="浅色主题", variable=self.theme_var, value="arc", command=self.change_theme)
        view_menu.add_radiobutton(label="深色主题", variable=self.theme_var, value="equilux", command=self.change_theme)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)

    def create_tooltip(self, widget, text):
        """创建工具提示"""
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            self.tooltip = Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            label = ttk.Label(self.tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1)
            label.pack()

        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def create_tooltips(self):
        """为所有控件创建工具提示"""
        # 为按钮创建工具提示
        self.create_tooltip(self.start_button, "开始训练过程")
        self.create_tooltip(self.pause_button, "暂停或恢复训练")
        self.create_tooltip(self.cancel_button, "取消训练过程")
        
        # 为下拉框创建工具提示
        self.create_tooltip(self.data_combo, "选择训练数据集")
        self.create_tooltip(self.model_combo, "选择训练模型")
        
        # 为输入框创建工具提示
        tooltips = {
            'batch_size': "每批处理的样本数量",
            'epochs': "训练轮数",
            'learning_rate': "学习率",
            'max_length': "输入序列的最大长度",
            'warmup_steps': "预热步数",
            'weight_decay': "权重衰减率",
            'gradient_accumulation_steps': "梯度累积步数"
        }
        
        for entry_name, tooltip_text in tooltips.items():
            if entry_name in self.entries:
                self.create_tooltip(self.entries[entry_name], tooltip_text)

    def browse_dataset(self):
        path = filedialog.askdirectory(initialdir=self.data_dir, title="Select Dataset Directory")
        if path:
            self.data_var.set(os.path.basename(path))
            self.update_data_list()

    def browse_model(self):
        path = filedialog.askdirectory(initialdir=self.models_dir, title="Select Model Directory")
        if path:
            self.model_var.set(os.path.basename(path))
            self.update_model_list()

    def change_theme(self):
        self.root.set_theme(self.theme_var.get())

    def show_error(self, title, message):
        messagebox.showerror(title, message)
        logging.error(f"Error displayed: {title} - {message}")

    def show_about(self):
        messagebox.showinfo("About", "Transformer Training Studio\nVersion 1.0\nDeveloped by xAI\n\nA tool for training transformer models with ease.")

    def view_logs(self):
        log_window = Toplevel(self.root)
        log_window.title("Training Logs")
        log_window.geometry("600x400")

        text_area = Text(log_window, wrap=tk.WORD, height=20, width=60)
        text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        scrollbar = Scrollbar(log_window, orient=tk.VERTICAL, command=text_area.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_area.config(yscrollcommand=scrollbar.set)

        try:
            with open("training.log", "r") as f:
                text_area.insert(tk.END, f.read())
        except FileNotFoundError:
            text_area.insert(tk.END, "No logs available.")
        text_area.config(state="disabled")

    def save_config(self):
        config = {
            "dataset": self.data_var.get(),
            "model": self.model_var.get(),
            "device": self.device_var.get(),
            "batch_size": self.batch_size_var.get(),
            "epochs": self.epochs_var.get(),
            "learning_rate": self.learning_rate_var.get(),
            "max_length": self.max_length_var.get(),
            "warmup_steps": self.warmup_steps_var.get(),
            "weight_decay": self.weight_decay_var.get(),
            "gradient_accumulation_steps": self.gradient_accumulation_steps_var.get(),
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as f:
                json.dump(config, f, indent=4)
            messagebox.showinfo("Success", "Configuration saved successfully!")
            logging.info(f"Configuration saved to {file_path}")

    def load_config(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as f:
                config = json.load(f)
            self.data_var.set(config.get("dataset", ""))
            self.model_var.set(config.get("model", ""))
            self.device_var.set(config.get("device", "CPU"))
            self.batch_size_var.set(config.get("batch_size", "1"))
            self.epochs_var.set(config.get("epochs", "1"))
            self.learning_rate_var.set(config.get("learning_rate", "2e-5"))
            self.max_length_var.set(config.get("max_length", "32"))
            self.warmup_steps_var.set(config.get("warmup_steps", "0"))
            self.weight_decay_var.set(config.get("weight_decay", "0.01"))
            self.gradient_accumulation_steps_var.set(config.get("gradient_accumulation_steps", "1"))
            self.update_data_list()
            self.update_model_list()
            messagebox.showinfo("Success", "Configuration loaded successfully!")
            logging.info(f"Configuration loaded from {file_path}")

    def update_data_list(self):
        if self.data_combo is None:
            return
        datasets = [item for item in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, item)) or item.endswith(('.jsonl', '.json', '.csv'))]
        self.data_combo['values'] = datasets
        if datasets and not self.data_var.get():
            self.data_var.set(datasets[0])
        self.estimate_memory()

    def update_model_list(self):
        if self.model_combo is None:
            return
        models = [item for item in os.listdir(self.models_dir) if os.path.isdir(os.path.join(self.models_dir, item))]
        self.model_combo['values'] = models
        if models and not self.model_var.get():
            self.model_var.set(models[0])
        self.estimate_memory()

    def estimate_memory(self):
        if self.ram_label is None or self.vram_label is None:
            return
        if not self.data_var.get() or not self.model_var.get():
            return

        try:
            data_path = os.path.join(self.data_dir, self.data_var.get())
            model_path = os.path.join(self.models_dir, self.model_var.get())

            model_size = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(model_path) for f in files) / (1024 * 1024) if os.path.exists(model_path) else 0
            dataset_size = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(data_path) for f in files) / (1024 * 1024) if os.path.isdir(data_path) else os.path.getsize(data_path) / (1024 * 1024) if os.path.exists(data_path) else 0

            total_ram = (model_size + dataset_size + 50) * 0.5
            total_vram = model_size * 0.1

            self.ram_label.config(text=f"{total_ram:.2f} MB")
            self.vram_label.config(text=f"{total_vram:.2f} MB")

            available_ram = psutil.virtual_memory().available / (1024 * 1024)
            if self.device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                available_vram = pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024 * 1024)
                if total_vram > available_vram or total_ram > available_ram:
                    self.show_error("Warning", "Insufficient memory detected, training may fail!")
                    logging.warning(f"Insufficient memory: VRAM needed {total_vram:.2f} MB, available {available_vram:.2f} MB; RAM needed {total_ram:.2f} MB, available {available_ram:.2f} MB")
        except Exception as e:
            logging.error(f"Error in estimate_memory: {str(e)}\n{traceback.format_exc()}")
            self.show_error("Error", f"Failed to estimate memory: {str(e)}")

    def start_training(self):
        if not self.data_var.get() or not self.model_var.get():
            self.show_error("Error", "Please select both a dataset and a model")
            logging.error("Training aborted: Dataset or model not selected")
            return

        try:
            batch_size = int(self.batch_size_var.get())
            epochs = int(self.epochs_var.get())
            learning_rate = float(self.learning_rate_var.get())
            if batch_size <= 0 or epochs <= 0 or learning_rate <= 0:
                raise ValueError("Batch size, epochs, and learning rate must be positive numbers.")
        except ValueError as e:
            self.show_error("Error", f"Invalid input: {str(e)}")
            logging.error(f"Invalid input: {str(e)}\n{traceback.format_exc()}")
            return

        self.is_training = True
        self.is_paused = False
        self.start_button.config(state="disabled")
        self.pause_button.config(state="normal", text="Pause")
        self.cancel_button.config(state="normal")
        self.status.config(text="Training in progress...")
        self.progress['value'] = 0
        self.training_thread = Thread(target=self.train_model)
        self.training_thread.start()
        logging.info("Training started")

    def pause_training(self):
        if not self.is_training:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.config(text="Resume")
            self.status.config(text="Training paused...")
            logging.info("Training paused")
        else:
            self.pause_button.config(text="Pause")
            self.status.config(text="Training in progress...")
            logging.info("Training resumed")

    def cancel_training(self):
        if not self.is_training:
            return
        if messagebox.askyesno("Confirm", "Are you sure you want to cancel the training?"):
            self.is_training = False
            self.is_paused = False
            self.start_button.config(state="normal")
            self.pause_button.config(state="disabled", text="Pause")
            self.cancel_button.config(state="disabled")
            self.status.config(text="Training cancelled.")
            self.progress['value'] = 0
            logging.info("Training cancelled by user")

    def on_closing(self):
        if self.is_training:
            if not messagebox.askyesno("Confirm", "Training is in progress. Do you want to exit?"):
                return
            self.is_training = False
            logging.info("Application closed during training")
        self.root.destroy()

    def clean_memory(self):
        """Clean up memory before and after training."""
        logging.debug("Cleaning VRAM and RAM")
        gc.collect()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            vram_free = torch.cuda.memory_reserved() / (1024 * 1024)
            logging.debug(f"VRAM free after cleanup: {vram_free:.2f} MiB")
        ram_free = psutil.virtual_memory().available / (1024 * 1024)
        logging.debug(f"RAM free after cleanup: {ram_free:.2f} MiB")

    def log_memory_usage(self, stage=""):
        """Log current memory usage."""
        ram_used = psutil.virtual_memory().used / (1024 * 1024)
        ram_free = psutil.virtual_memory().available / (1024 * 1024)
        logging.debug(f"RAM usage at {stage}: {ram_used:.2f} MiB used, {ram_free:.2f} MiB free")
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024 * 1024)
            vram_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            logging.debug(f"VRAM usage at {stage}: {vram_used:.2f} MiB allocated, {vram_reserved:.2f} MiB reserved")

    def train_model(self):
        """训练模型"""
        try:
            # 设置设备
            device = torch.device("cuda")
            torch.cuda.set_per_process_memory_fraction(0.8)  # 限制GPU内存使用
            
            # 获取模型和数据路径
            model_path = os.path.join(self.models_dir, self.model_var.get())
            data_path = os.path.join(self.data_dir, self.data_var.get())
            
            logging.info(f"使用模型路径: {model_path}")
            logging.info(f"使用数据路径: {data_path}")
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="cuda",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # 确保模型在CUDA上
            model = model.cuda()
            
            # 配置LoRA
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # 应用LoRA
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # 确保LoRA参数需要梯度
            for name, param in model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
            
            # 启用梯度检查点以节省内存
            model.gradient_checkpointing_enable()
            
            # 加载数据集
            dataset = CustomDataset(
                data_path,
                tokenizer,
                max_length=32
            )
            
            # 配置数据整理器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # 配置训练参数
            training_args = TrainingArguments(
                output_dir=self.trained_models_dir,
                num_train_epochs=int(self.epochs_var.get()),
                per_device_train_batch_size=int(self.batch_size_var.get()),
                gradient_accumulation_steps=4,
                learning_rate=float(self.learning_rate_var.get()),
                weight_decay=0.01,
                warmup_steps=100,
                logging_steps=10,
                save_strategy="epoch",
                evaluation_strategy="no",
                fp16=True,
                no_cuda=False,
                dataloader_pin_memory=True,
                remove_unused_columns=False,
                report_to="none"
            )
            
            # 创建训练器
            self.trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator
            )
            
            # 开始训练
            self.trainer.train(resume_from_checkpoint=None)
            
            # 保存模型
            self.trainer.save_model()
            logging.info("模型训练完成并保存")
            
        except Exception as e:
            logging.error(f"训练过程中出错: {str(e)}")
            raise

    def preview_dataset(self):
        """预览数据集内容"""
        if not self.data_var.get():
            self.show_error("错误", "请先选择数据集")
            return

        data_path = os.path.join(self.data_dir, self.data_var.get())
        if not os.path.exists(data_path):
            self.show_error("错误", "数据集不存在")
            return

        preview_window = Toplevel(self.root)
        preview_window.title("数据集预览")
        preview_window.geometry("800x600")

        # 创建预览区域
        preview_frame = ttk.Frame(preview_window, padding="10")
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_window.columnconfigure(0, weight=1)
        preview_window.rowconfigure(0, weight=1)

        # 创建文本区域
        text_area = Text(preview_frame, wrap=tk.WORD, height=20, width=80)
        text_area.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=text_area.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text_area.config(yscrollcommand=scrollbar.set)

        try:
            if os.path.isdir(data_path):
                text_path = os.path.join(data_path, "text")
                if os.path.exists(text_path):
                    files = os.listdir(text_path)[:5]  # 只显示前5个文件
                    for file in files:
                        if file.endswith('.txt'):
                            with open(os.path.join(text_path, file), 'r', encoding='utf-8') as f:
                                text_area.insert(tk.END, f"文件: {file}\n")
                                text_area.insert(tk.END, f.read() + "\n\n")
            elif os.path.isfile(data_path):
                if data_path.endswith('.jsonl'):
                    with open(data_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if i >= 5:  # 只显示前5条记录
                                break
                            item = json.loads(line.strip())
                            text_area.insert(tk.END, f"记录 {i+1}:\n")
                            text_area.insert(tk.END, json.dumps(item, indent=2, ensure_ascii=False) + "\n\n")
                elif data_path.endswith('.json'):
                    with open(data_path, 'r', encoding='utf-8') as f:
                        items = json.load(f)
                        for i, item in enumerate(items[:5]):  # 只显示前5条记录
                            text_area.insert(tk.END, f"记录 {i+1}:\n")
                            text_area.insert(tk.END, json.dumps(item, indent=2, ensure_ascii=False) + "\n\n")
                elif data_path.endswith('.csv'):
                    df = pd.read_csv(data_path)
                    text_area.insert(tk.END, "数据集信息:\n")
                    text_area.insert(tk.END, f"总行数: {len(df)}\n")
                    text_area.insert(tk.END, f"列名: {', '.join(df.columns)}\n\n")
                    text_area.insert(tk.END, "前5条记录:\n")
                    text_area.insert(tk.END, df.head().to_string() + "\n")

            text_area.config(state="disabled")
        except Exception as e:
            self.show_error("错误", f"预览数据集时出错: {str(e)}")
            logging.error(f"预览数据集时出错: {str(e)}\n{traceback.format_exc()}")

    def convert_dataset(self):
        """转换数据集格式"""
        if not self.data_var.get():
            self.show_error("错误", "请先选择数据集")
            return

        source_path = os.path.join(self.data_dir, self.data_var.get())
        if not os.path.exists(source_path):
            self.show_error("错误", "数据集不存在")
            return

        target_format = self.target_format_var.get()
        target_path = os.path.join(self.data_dir, f"{os.path.splitext(self.data_var.get())[0]}.{target_format}")

        try:
            if os.path.isdir(source_path):
                # 从目录格式转换
                text_path = os.path.join(source_path, "text")
                image_path = os.path.join(source_path, "images")
                if not os.path.exists(text_path):
                    raise ValueError("目录格式必须包含text子目录")

                data = []
                for file in os.listdir(text_path):
                    if file.endswith('.txt'):
                        with open(os.path.join(text_path, file), 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        if text:
                            image_file = os.path.join(image_path, file.replace('.txt', '.jpg'))
                            data.append({
                                'text': text,
                                'image': image_file if os.path.exists(image_file) else None
                            })
            elif os.path.isfile(source_path):
                # 从其他格式转换
                if source_path.endswith('.jsonl'):
                    with open(source_path, 'r', encoding='utf-8') as f:
                        data = [json.loads(line.strip()) for line in f]
                elif source_path.endswith('.json'):
                    with open(source_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                elif source_path.endswith('.csv'):
                    df = pd.read_csv(source_path)
                    data = df.to_dict('records')
                else:
                    raise ValueError(f"不支持的数据集格式: {os.path.splitext(source_path)[1]}")

            # 保存为目标格式
            if target_format == 'jsonl':
                with open(target_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            elif target_format == 'json':
                with open(target_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            elif target_format == 'csv':
                df = pd.DataFrame(data)
                df.to_csv(target_path, index=False, encoding='utf-8')

            messagebox.showinfo("成功", f"数据集已转换为{target_format.upper()}格式")
            logging.info(f"数据集已转换为{target_format.upper()}格式: {target_path}")
            self.update_data_list()
        except Exception as e:
            self.show_error("错误", f"转换数据集时出错: {str(e)}")
            logging.error(f"转换数据集时出错: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    try:
        # 首先运行依赖检查
        from check_dependencies import main as check_dependencies
        if not check_dependencies():
            logging.critical("依赖检查失败，程序无法启动")
            sys.exit(1)
            
        # 依赖检查通过后启动主程序
        root = ThemedTk(theme="arc")
        app = TrainingApp(root)
        root.mainloop()
    except Exception as e:
        logging.critical(f"程序启动失败: {str(e)}\n{traceback.format_exc()}")
        raise