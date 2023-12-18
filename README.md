# FinSA基本介绍
我们设计并实现了一个叫做FinSA的金融情感分析代码框架，
名称取自**Fin**ancial **S**entiment **A**nalysis的缩写。
该框架能够支持包括FinBERT，FinGPT在内的各个模型
在金融情感分析领域的几个常见数据集上的训练和推理。
该框架具有可扩展性，采用了统一的处理流程，参数、返回值接口，配置管理等设计。

# FinSA结构设计
该项目的代码结构如下：
```
FinSA
│  
│  run_finsa_inference.py
│
├─configs
│  │  __init__.py
│  │  
│  ├─dataset_configs
│  │      base_dataset_config.py
│  │      fiqa_config.py
│  │      fpb_config.py
│  │      nwgi_config.py
│  │      tfns_config.py
│  │      __init__.py
│  │      
│  └─model_configs
│          base_model_config.py
│          bert_config.py
│          chatglm_model_config.py
│          llama_family_model_config.py
│          __init__.py
│          
├─datasets
│  │    
│  ├─financial_phrasebank-sentences_50agree
│  │          
│  ├─fiqa-2018
│  │          
│  ├─news_with_gpt_instructions
│  │          
│  └─twitter-financial-news-sentiment
│              
└─utils
    │  __init__.py
    │  
    ├─config_utils
    │      config_utils.py
    │      __init__.py
    │      
    ├─dataset_utils
    │      fiqa_utils.py
    │      fpb_utils.py
    │      metric_utils.py
    │      nwgi_utils.py
    │      tfns_utils.py
    │      __init__.py
    │      
    └─model_utils
            inference_utils.py
            __init__.py
            

```
其中：
1. FinSA/datsets为数据目录，里面包含了四个数据集（简称为FPB、FIQA、NSGI和TFNS），
可以通过datasets库进行load，例如FIQA数据集的载入与处理可以详见FinSA/utils/dataset_utils/fiqa_utils.py。
2. FinSA/dataset_configs目录中是相关的Dataset Config类，
FinSA/model_configs目录中是相关的Model Config类，具体由python dataclass工具类实现。
   1. base_model_config.py中定义了BaseModelConfig基类，
   里面是各模型共有的配置项，例如model_name，model_path，padding_side等。
   其余的xxx_model_config.py里定义了XXXModelConfig派生类，里面是具体的模型设置以及可能的特有的新配置项，
   例如FinBERT模型需要大小写转换，因此其对应的BertConfig中新增添了特有的配置项do_lower_case。
   2. 类似的，base_dataset_config.py中定义了BaseDatasetConfig基类，
   里面是各数据集共有的配置项，例如dataset_name，dataset_name，batch_size等。
   其余的xxx_dataset_config.py里定义了XXXModelConfig派生类。
   3. 上述设计具有可扩展性，用户如果需要引入新的自定义数据集或自定义模型，只需依照示例
   声明自定义的Dataset Config或Model Config，并继承基类即可。
3. FinSA/run_finsa_inference.py为各模型在上述四个数据集上进行推理测试的入口，
请仔细阅读，简单的示例的运行命令（其他超参数都由FinSA/configs里对应的Config默认给定配置，
也可以通过命令行进行更高优先级的覆盖，但默认值已经精心设置，如无必要请勿改动）如下：
```
# 运行FinGPT（以FinGPT v3.2为例）
python run_finsa_inference.py --model_name fingpt-v3.2 --datasets fpb,fiqa,tfns,nwgi
# 运行FinBERT
python run_finsa_inference.py --model_name finbert --datasets fpb,fiqa,tfns,nwgi
# 运行其他baseline（以Alpaca为例）
python run_finsa_inference.py --model_name alpaca --datasets fpb,fiqa,tfns,nwgi
```
4. FinSA/utils里是工具函数，其中：
    1. config_utils里实现了通过命令行传入的kwargs更新各个Config的工具函数update_configs。
    2. model_utils里实现了由run_finsa_inference.py调用的prepare_tokenizer_and_model_inference，
   它根据model_name来分别调用fingpt或finbert的推理，具有可扩展性，用户可以仿照示例根据自己的需求
   自行更改推理时的具体实现。
    3. datset_utils里包含了metric_utils.py和各个数据集的dataloader的构造代码，对应后处理和前处理。
   其设计同样具有可扩展性，用户可自行修改。

# FinSA实验结果
以下为我们使用FinSA框架进行金融情感分析的实验结果，Metric选用Weighted-F1值。

![FinSA实验结果](./results/FinSA实验结果.pdf)