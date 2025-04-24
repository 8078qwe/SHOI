
from mmdet.models import HEADS, StandardRoIHead


@HEADS.register_module()
class RoIHead(StandardRoIHead):
    def __init__(self, param, **kwargs):
        super().__init__(**kwargs)
        self.param = self.param
