"""Tokenization classes for Yak."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ...tokenization_utils_base import TextInput

from ..llama import LlamaTokenizer


SPIECE_UNDERLINE = "▁"

class YakTokenizer(LlamaTokenizer):

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        use_default_system_prompt=False,
        spaces_between_special_tokens=False,
        legacy=False,
        add_prefix_space=True,
        **kwargs,
    ):
        # Same as LlamaTokenizer except default legacy=False.
        super().__init__(
            vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sp_model_kwargs=sp_model_kwargs,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            use_default_system_prompt=use_default_system_prompt,
            spaces_between_special_tokens=spaces_between_special_tokens,
            legacy=legacy,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    def tokenize(self, text: "TextInput", **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the
        first token is special.

        This implementation is a hack to work around a bug in the Llama implementation, which does not
        check the _additional_special_tokens field when determining whether to remove the SPIECE_UNDERLINE
        prefix from the text before tokenization.
        """
        tokens = super().tokenize(text, **kwargs)

        # if the second token is in _additional_special_tokens,
        # then remove the extraneous SPIECE_UNDERLINE prefix added by the parent class
        if len(tokens) >= 2 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self._additional_special_tokens:
            tokens = tokens[1:]

        return tokens

    @property
    def default_chat_template(self):
        """
        This template formats inputs in the standard Yak format.
        """
        return (
            "{% for message in messages %}"
            "{{'<s>' + message['role'] + '\n' + message['content'] + '</s>'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<s>assistant\n' }}"
            "{% endif %}"
        )
