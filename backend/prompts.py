
class PromptFormat:

    def __init__(self):
        pass

    def format(self, prompt, response, system_prompt, settings):
        raise NotImplementedError

    def stop_conditions(self, tokenizer, settings):
        raise NotImplementedError

    def is_instruct(self):
        raise NotImplementedError

    def encode_special_tokens(self):
        return True


class PromptFormat_raw(PromptFormat):

    description = "Model-agnostic mode simulating a raw chatlog between two or more users"

    def __init__(self):
        super().__init__()
        pass

    def is_instruct(self):
        return False

    def stop_conditions(self, tokenizer, settings):
        raise NotImplementedError

    def format(self, prompt, response, system_prompt, settings):
        raise NotImplementedError

    def encode_special_tokens(self):
        return False


class PromptFormat_llama(PromptFormat):

    description = "Llama-chat, Llama2-chat and Mistral-instruct models"

    def __init__(self):
        super().__init__()
        pass

    def is_instruct(self):
        return True

    def stop_conditions(self, tokenizer, settings):
        return \
            [tokenizer.eos_token_id]

    def format(self, prompt, response, system_prompt, settings):
        text = "<s>[INST] "
        if system_prompt and system_prompt.strip() != "":
            text += "<<SYS>>\n"
            text += system_prompt
            text += "\n<</SYS>>\n\n "
        text += prompt
        text += " [/INST]"
        if response:
            text += response
            text += "</s>"
        return text


class PromptFormat_mistrallite(PromptFormat):

    description = "MistralLite format"

    def __init__(self):
        super().__init__()
        pass

    def is_instruct(self):
        return True

    def stop_conditions(self, tokenizer, settings):
        return \
            [tokenizer.eos_token_id]

    def format(self, prompt, response, system_prompt, settings):
        text = "<|prompter|>"
        if system_prompt and system_prompt.strip() != "":
            text += system_prompt
            text += "</s><|assistant|>Understood.</s><|prompter|>"
        text += prompt
        text += "</s><|assistant|>"
        if response:
            text += response
            text += "</s>"
        return text

# class PromptFormat_codellama(PromptFormat_llama):
#
#     description = "CodeLlama-instruct"
#
#     def __init__(self):
#         super().__init__()
#         pass
#
#     def default_system_prompt(self):
#         return \
#             """You are a helpful coding assistant. Always answer as helpfully as possible."""


class PromptFormat_chatml(PromptFormat):

    description = "ChatML format, as used by e.g. (Mistral)Orca"

    def __init__(self):
        super().__init__()
        pass

    def is_instruct(self):
        return True

    def stop_conditions(self, tokenizer, settings):
        return \
            [tokenizer.eos_token_id,
             """<|im_end|>"""]

    def format(self, prompt, response, system_prompt, settings):
        text = ""
        if system_prompt and system_prompt.strip() != "":
            text += "<|im_start|>system\n"
            text += system_prompt
            text += "\n<|im_end|>\n"
        text += "<|im_start|>user\n"
        text += prompt
        text += "<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        if response:
            text += response
            text += "<|im_end|>\n"
        return text


class PromptFormat_tinyllama(PromptFormat_chatml):

    description = "ChatML format, but ignoring special/added tokens. Use for TinyLlama-chat v0.3"

    def encode_special_tokens(self):
        return False


class PromptFormat_phind_codellama(PromptFormat):

    description = "Vicuna/Alpaca-like format for Phind-CodeLlama"

    def __init__(self):
        super().__init__()
        pass

    def is_instruct(self):
        return True

    def stop_conditions(self, tokenizer, settings):
        return \
            [tokenizer.eos_token_id, "\n### "]

    def format(self, prompt, response, system_prompt, settings):
        text = ""
        if system_prompt and system_prompt.strip() != "":
            text += "### System Prompt\n"
            text += system_prompt
            text += "\n\n"
        text += "### User Message\n"
        text += prompt
        text += "\n\n### Assistant\n"
        if response:
            text += response
            text += "\n\n"
        return text


class PromptFormat_deepseek_chat(PromptFormat):

    description = "Deepseek LLM chat format"

    def __init__(self):
        super().__init__()
        pass

    def is_instruct(self):
        return True

    def stop_conditions(self, tokenizer, settings):
        return \
            [tokenizer.eos_token_id, "\n\nAssistant:"]

    def format(self, prompt, response, system_prompt, settings):
        text = ""
        if system_prompt and system_prompt.strip() != "":
            text += system_prompt
            text += "\n\n"
        text += "User: "
        text += prompt
        text += "\n\nAssistant:"
        if response:
            text += response
            text += "\n\n"
        return text


class PromptFormat_deepseek_instruct(PromptFormat):

    description = "Deepseek instruct format for 'coder' models"

    def __init__(self):
        super().__init__()
        pass

    def is_instruct(self):
        return True

    def stop_conditions(self, tokenizer, settings):
        return \
            [tokenizer.eos_token_id, "<|EOT|>"]

    def format(self, prompt, response, system_prompt, settings):
        text = ""
        if system_prompt and system_prompt.strip() != "":
            text += "<｜begin▁of▁sentence｜>"
            text += system_prompt
            text += "\n"
        text += "### Instruction:\n"
        text += prompt
        text += "\n### Response:\n"
        if response:
            text += response
            text += "\n<|EOT|>\n"
        return text



prompt_formats = \
{
    "Chat-RP": PromptFormat_raw,
    "Llama-chat": PromptFormat_llama,
    "ChatML": PromptFormat_chatml,
    "TinyLlama-chat": PromptFormat_tinyllama,
    "MistralLite": PromptFormat_mistrallite,
    "Phind-CodeLlama": PromptFormat_phind_codellama,
    "Deepseek-chat": PromptFormat_deepseek_chat,
    "Deepseek-instruct": PromptFormat_deepseek_instruct,
}

def list_prompt_formats():
    global prompt_formats
    return list(prompt_formats.keys())
