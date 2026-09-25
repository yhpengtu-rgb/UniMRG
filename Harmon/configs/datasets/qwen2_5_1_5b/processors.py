from xtuner.utils import PROMPT_TEMPLATE
from src.models.dllm.mask_token import load_harmon_tokenizer


llm_name_or_path = __import__('os').environ.get(
    'HARMON_LLM_PATH',
    '/nvmedata/xiexu/data/uni/Qwen2.5-1.5B-Instruct',
)
prompt_template = PROMPT_TEMPLATE.qwen_chat
pad_index = 151645
image_length = 1024 + 64
image_size = 512

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=load_harmon_tokenizer,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right',
    local_files_only=True)
