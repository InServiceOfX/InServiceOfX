class Statement(dict):
    def __init__(self, role, content=""):
        super().__init__(role=role, content=content)

    def to_dict(self):
        return {"role": self["role"], "content": self["content"]}

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"


class SystemStatement(Statement):
    def __init__(self, content=""):
        super().__init__("system", content)


class UserStatement(Statement):
    def __init__(self, content=""):
        super().__init__("user", content)


class AssistantStatement(Statement):
    def __init__(self, content=""):
        super().__init__("assistant", content)


