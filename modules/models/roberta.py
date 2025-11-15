from transformers import RobertaConfig, AutoConfig
from transformers import RobertaForSequenceClassification
from modules.models.parameter_efficient_llm import PELLM


class Roberta(PELLM):
    config_class = RobertaConfig
    model_loader = RobertaForSequenceClassification

    def __init__(self, config: dict = None,
                 pretrained_path: str = None,
                 peft_type: str = None,
                 peft_config: dict = None,
                 **kwargs) -> None:

        if pretrained_path is not None:
            self.check_config(pretrain_path=pretrained_path)
        if config is None and pretrained_path is None:
            config = RobertaConfig().to_dict()
        super().__init__(
            config=config,
            pretrained_path=pretrained_path,
            peft_type=peft_type,
            peft_config=peft_config,
            **kwargs)

    def check_config(self, pretrain_path):
        config = AutoConfig.from_pretrained(pretrain_path)
        assert isinstance(
            config, RobertaConfig), 'The config of pretrained model must be RobertaConfig, but got {}'.format(
            type(config))