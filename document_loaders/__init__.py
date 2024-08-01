import importlib
import logging
import os

import yaml

CONFIG_FILE = "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/document_loaders/config.yaml"


class DocumentLoaderConfig:
    def __init__(self, config_file: str):
        try:
            with open(config_file) as f:
                self._yaml_data = yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Config file {config_file} not found, using default config")
            self._yaml_data = {
                "loader": [
                    {
                        "extension": "default",
                        "class": "UnstructuredFileLoader",
                        "module": "langchain.document_loaders.unstructured",
                    }
                ]
            }
        self._build_loaders()

    @property
    def document_loaders(self):
        return self._document_loaders

    def _init_component(self, component: dict):
        env_variables = component.get("env", {})
        for key, value in env_variables.items():
            os.environ[key] = value
        shell_commands = component.get("shell", [])
        for command in shell_commands:
            os.system(command)

    def _build_loaders(self):
        loader_map = {}
        if "loader" not in self._yaml_data:
            raise ValueError("No loaders found in config file")
        for loader in self._yaml_data["loader"]:
            self._init_component(loader)
            extension = loader.get("extension", None)
            if extension is None:
                logging.error(
                    f"No extension found for loader {loader}"
                )
                continue
            module_name = loader.get("module", None)
            if module_name is None:
                logging.error(
                    f"No module found for loader {loader}"
                )
                continue
            class_name = loader.get("class", None)
            if class_name is None:
                logging.error(
                    f"No class found for loader {loader}"
                )
                continue
            try:
                loader_module = importlib.import_module(module_name)
                loader_class = getattr(loader_module, class_name)
            except ImportError as e:
                logging.error(
                    f"Error importing loader {loader}: {e}"
                )
                continue
            loader_kwargs = loader.get("params", {})
            if extension in loader_map:
                logging.warning(
                    f"Overriding loader for extension {extension}"
                )
            loader_map[extension] = {"class": loader_class, "kwargs": loader_kwargs}
        self._document_loaders = loader_map


loaders_config = DocumentLoaderConfig(CONFIG_FILE)
