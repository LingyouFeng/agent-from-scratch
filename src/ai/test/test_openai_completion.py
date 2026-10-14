import time
import unittest

from ai.model import Model
from agent import stream
from agent import (
    Context,
    TextContent,
    UserMessage,
)


class TestStreamChat(unittest.TestCase):
    def test_real_stream_call(self):
        """
        Uncomment to run real network test.
        Replace api_key with your valid key
        """
        test_model = Model(
            id="deepseek-v4-flash",
            name="deepseek-v4-flash",
            provider="deepseek",
            api="openai-completions",
        )
        user_msg = UserMessage(
            role="user",
            content=[TextContent(type="text", text="推荐青岛餐厅，适合父母")],
            timestamp=int(time.time() * 1000)
        )
        ctx = Context(
            systemPrompt="回答简洁",
            messages=[user_msg],
            tools=[]
        )
        print("\n=== Start stream ===")
        resp_msg = stream(test_model, ctx)
        print("\n=== Done ===")
        self.assertEqual(resp_msg.role, "assistant")


if __name__ == "__main__":
    unittest.main()
