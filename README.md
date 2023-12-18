# FinSA��������
������Ʋ�ʵ����һ������FinSA�Ľ�����з��������ܣ�
����ȡ��**Fin**ancial **S**entiment **A**nalysis����д��
�ÿ���ܹ�֧�ְ���FinBERT��FinGPT���ڵĸ���ģ��
�ڽ�����з�������ļ����������ݼ��ϵ�ѵ��������
�ÿ�ܾ��п���չ�ԣ�������ͳһ�Ĵ������̣�����������ֵ�ӿڣ����ù������ơ�

# FinSA�ṹ���
����Ŀ�Ĵ���ṹ���£�
```
FinSA
��  
��  run_finsa_inference.py
��
����configs
��  ��  __init__.py
��  ��  
��  ����dataset_configs
��  ��      base_dataset_config.py
��  ��      fiqa_config.py
��  ��      fpb_config.py
��  ��      nwgi_config.py
��  ��      tfns_config.py
��  ��      __init__.py
��  ��      
��  ����model_configs
��          base_model_config.py
��          bert_config.py
��          chatglm_model_config.py
��          llama_family_model_config.py
��          __init__.py
��          
����datasets
��  ��    
��  ����financial_phrasebank-sentences_50agree
��  ��          
��  ����fiqa-2018
��  ��          
��  ����news_with_gpt_instructions
��  ��          
��  ����twitter-financial-news-sentiment
��              
����utils
    ��  __init__.py
    ��  
    ����config_utils
    ��      config_utils.py
    ��      __init__.py
    ��      
    ����dataset_utils
    ��      fiqa_utils.py
    ��      fpb_utils.py
    ��      metric_utils.py
    ��      nwgi_utils.py
    ��      tfns_utils.py
    ��      __init__.py
    ��      
    ����model_utils
            inference_utils.py
            __init__.py
            

```
���У�
1. FinSA/datsetsΪ����Ŀ¼������������ĸ����ݼ������ΪFPB��FIQA��NSGI��TFNS����
����ͨ��datasets�����load������FIQA���ݼ��������봦��������FinSA/utils/dataset_utils/fiqa_utils.py��
2. FinSA/dataset_configsĿ¼������ص�Dataset Config�࣬
FinSA/model_configsĿ¼������ص�Model Config�࣬������python dataclass������ʵ�֡�
   1. base_model_config.py�ж�����BaseModelConfig���࣬
   �����Ǹ�ģ�͹��е����������model_name��model_path��padding_side�ȡ�
   �����xxx_model_config.py�ﶨ����XXXModelConfig�����࣬�����Ǿ����ģ�������Լ����ܵ����е��������
   ����FinBERTģ����Ҫ��Сдת����������Ӧ��BertConfig�������������е�������do_lower_case��
   2. ���Ƶģ�base_dataset_config.py�ж�����BaseDatasetConfig���࣬
   �����Ǹ����ݼ����е����������dataset_name��dataset_name��batch_size�ȡ�
   �����xxx_dataset_config.py�ﶨ����XXXModelConfig�����ࡣ
   3. ������ƾ��п���չ�ԣ��û������Ҫ�����µ��Զ������ݼ����Զ���ģ�ͣ�ֻ������ʾ��
   �����Զ����Dataset Config��Model Config�����̳л��༴�ɡ�
3. FinSA/run_finsa_inference.pyΪ��ģ���������ĸ����ݼ��Ͻ���������Ե���ڣ�
����ϸ�Ķ����򵥵�ʾ�������������������������FinSA/configs���Ӧ��ConfigĬ�ϸ������ã�
Ҳ����ͨ�������н��и������ȼ��ĸ��ǣ���Ĭ��ֵ�Ѿ��������ã����ޱ�Ҫ����Ķ������£�
```
# ����FinGPT����FinGPT v3.2Ϊ����
python run_finsa_inference.py --model_name fingpt-v3.2 --datasets fpb,fiqa,tfns,nwgi
# ����FinBERT
python run_finsa_inference.py --model_name finbert --datasets fpb,fiqa,tfns,nwgi
# ��������baseline����AlpacaΪ����
python run_finsa_inference.py --model_name alpaca --datasets fpb,fiqa,tfns,nwgi
```
4. FinSA/utils���ǹ��ߺ��������У�
    1. config_utils��ʵ����ͨ�������д����kwargs���¸���Config�Ĺ��ߺ���update_configs��
    2. model_utils��ʵ������run_finsa_inference.py���õ�prepare_tokenizer_and_model_inference��
   ������model_name���ֱ����fingpt��finbert���������п���չ�ԣ��û����Է���ʾ�������Լ�������
   ���и�������ʱ�ľ���ʵ�֡�
    3. datset_utils�������metric_utils.py�͸������ݼ���dataloader�Ĺ�����룬��Ӧ�����ǰ����
   �����ͬ�����п���չ�ԣ��û��������޸ġ�

# FinSAʵ����
����Ϊ����ʹ��FinSA��ܽ��н�����з�����ʵ������Metricѡ��Weighted-F1ֵ��

![FinSAʵ����](./results/FinSAʵ����.pdf)