"""ZhipuAI chat wrapper."""
from __future__ import annotations

import json
import logging
from typing import List, Optional, Any, Dict

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema import BaseMessage, ChatResult, AIMessage, ChatGeneration
from langchain.schema.messages import FunctionMessage, HumanMessage, ChatMessage, SystemMessage
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a message to a dictionary that can be passed to the API."""
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict


class ChatZhipuAI(BaseChatModel):
    """`ZhipuAI` Chat large language models API.

        To use, you should have the ``zhipuai`` python package installed, and the
        environment variable ``ZHIPUAI_API_KEY`` set with your API key.

        Example:
            .. code-block:: python

                from chat_models import ChatZhipuAI
                zhipuai = ChatZhipuAI(model_name="chatglm_turbo")
        """

    client: Any = None  #: :meta private:
    model: str = "chatglm_turbo"
    """Model name to use."""
    temperature: Optional[float] = 0.95
    """What sampling temperature to use.
    Range in (0.0, 1.0]. Default: 0.95"""
    top_p: Optional[float] = 0.7
    """The cumulative probability threshold during sampling.
    Range in (0.0, 1.0). Default: 0.7"""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    zhipuai_api_key: Optional[str] = None

    request_timeout: Optional[int] = 60
    """request timeout for chat http requests"""
    incremental: Optional[bool] = True
    """In SSE interface calls, Control whether the content is returned on each occasion incrementally or in full.
    
    - True: incremental
    
    - False: full"""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"zhipuai_api_key": "ZHIPUAI_API_KEY"}

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling ZhipuAI API."""
        return {
            "model": self.model,
            "request_timeout": self.request_timeout,
            "incremental": self.incremental,
            "top_p": self.top_p,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat_model."""
        return "zhipuai-chat"

    def _convert_prompt_msg_params(
            self,
            messages: List[BaseMessage],
            **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Converts a list of messages into a dictionary containing the message content
        and default parameters.

        Args:
            messages (List[BaseMessage]): The list of messages.
            **kwargs (Any): Optional arguments to add additional parameters to the
            resulting dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the message content and default
            parameters.

        """
        messages_dict: Dict[str, Any] = {
            "prompt": [
                convert_message_to_dict(m)
                for m in messages
                if not isinstance(m, SystemMessage)
            ]
        }
        for i in [i for i, m in enumerate(messages) if isinstance(m, SystemMessage)]:
            if "system" not in messages_dict:
                messages_dict["system"] = ""
            messages_dict["system"] += messages[i].content + "\n"

        return {
            **messages_dict,
            **self._default_params,
            **kwargs,
        }

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any
    ) -> ChatResult:
        """Call out to an ZhipuAI models for each generation with a prompt.
        Args:
            messages: The messages to pass into the model
            stop: Optional list of stop words to use when generating
        Returns:
            The string generated by the model
        """
        params = self._convert_prompt_msg_params(messages, **kwargs)
        response_payload = self.client.sse_invoke(**params)
        completion = ""
        meta: Dict = {}
        # TODO: Support streaming
        if self.incremental:
            for event in response_payload.events():
                completion += event.data
                if event.event == "finish":
                    meta = json.loads(event.meta)
        else:
            for event in response_payload.events():
                if event.event == "finish":
                    completion = event.data
                    meta = json.loads(event.meta)
        lc_msg = AIMessage(content=completion, additional_kwargs=meta)
        gen = ChatGeneration(
            message=lc_msg,
            generation_info=dict(finish_reason="finish"),
        )
        return ChatResult(
            generations=[gen],
            llm_output={"model_name": self.model},
        )

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["zhipuai_api_key"] = get_from_dict_or_env(
            values,
            "zhipuai_api_key",
            "ZHIPUAI_API_KEY"
        )
        try:
            import zhipuai
            zhipuai.api_key = values["zhipuai_api_key"]
            values["client"] = zhipuai.model_api
        except ImportError:
            raise ValueError(
                "zhipuai package not found, please install it with "
                "`pip install zhipuai`"
            )
        return values
