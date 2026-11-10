"""复用现有配置；不把密钥写入 repr、报错或日志。"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    api_style: str = 'auto'
    root: Path = ROOT

    @classmethod
    def load(cls, root=ROOT):
        root = Path(root).resolve()
        values = {**dotenv_values(root.parent / '.env'),
                  **dotenv_values(root / '.env'), **os.environ}
        settings = cls(
            base_url=str(values.get('AGENT_BASE_URL') or '').strip().rstrip('/'),
            model=str(values.get('AGENT_MODEL') or '').strip(),
            api_key=str(values.get('AGENT_API_KEY') or '').strip(),
            api_style=str(values.get('AGENT_API_STYLE') or 'auto').strip(), root=root)
        settings.validate()
        return settings

    def validate(self):
        for name, value in (('AGENT_MODEL', self.model), ('AGENT_API_KEY', self.api_key)):
            if (not value or value.startswith(('your-', 'your_')) or '...' in value
                    or '…' in value or value.lower() in {'replace-me', 'changeme'}):
                raise ValueError(f'{name} 缺失或是占位符，请在本机 .env 中填写有效值。')
        url = urlsplit(self.base_url)
        local = url.hostname in {'localhost', '127.0.0.1', '::1'}
        if (not url.hostname or url.username or url.password or url.query or url.fragment
                or not (url.scheme == 'https' or (url.scheme == 'http' and local))):
            raise ValueError('AGENT_BASE_URL 必须为 HTTPS API 根地址，本机服务可使用 HTTP。')
        if self.api_style not in {'auto', 'responses', 'chat_completions'}:
            raise ValueError('AGENT_API_STYLE 应为 auto、responses 或 chat_completions。')

    @property
    def fingerprint(self):
        import hashlib
        return hashlib.sha256((self.base_url + '\n' + self.model).encode()).hexdigest()
