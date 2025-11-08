class BuildDockerBaseClass:
    def __init__(self, docker_image_name: str, base_image: str):
        self.docker_image_name = docker_image_name
        self.base_image = base_image

    def build(self):
        pass