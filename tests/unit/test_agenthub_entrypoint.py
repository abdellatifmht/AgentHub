import unittest

import agenthub


class AgentHubEntrypointTest(unittest.TestCase):
    def test_main_entrypoint_is_exposed(self) -> None:
        self.assertTrue(callable(agenthub.main))


if __name__ == "__main__":
    unittest.main()
