"""
Stage 6 Docker/部署相关测试。

不实际跑 docker build（CI 复杂、网络依赖 Docker Hub），而是：
- 静态校验 Dockerfile / compose 文件存在且语法正确
- 校验 .dockerignore 正确排除敏感文件
- 校验配置文件可被相关工具加载
"""

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# 必需文件存在
# ---------------------------------------------------------------------------

class TestStage6Files:
    """所有 Stage 6 新增的部署文件都应该存在。"""

    @pytest.mark.parametrize("filename", [
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        "docker-entrypoint.sh",
        "gunicorn.conf.py",
        ".env.example",
        "README.md",
    ])
    def test_file_exists(self, filename):
        path = PROJECT_ROOT / filename
        assert path.exists(), f"缺少部署文件: {filename}"

    def test_entrypoint_is_executable(self):
        """entrypoint 脚本必须有可执行权限。"""
        path = PROJECT_ROOT / "docker-entrypoint.sh"
        mode = path.stat().st_mode
        assert mode & 0o100, "docker-entrypoint.sh 缺少可执行权限"


# ---------------------------------------------------------------------------
# 密钥安全：.dockerignore 必须排除敏感文件
# ---------------------------------------------------------------------------

class TestSecretSafety:
    """关键：API key 等敏感信息绝不能进 build context。"""

    def test_env_excluded_from_context(self):
        """`.env` 必须在 .dockerignore 里。"""
        content = (PROJECT_ROOT / ".dockerignore").read_text()
        assert ".env" in content

    def test_env_not_in_dockerfile_copy(self):
        """Dockerfile 不能 COPY .env。"""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        # 所有 COPY 命令
        copy_lines = [
            line for line in content.split("\n") if line.strip().startswith("COPY")
        ]
        for line in copy_lines:
            # 不能含 .env（.env.example 是 OK 的）
            assert ".env" not in line or ".env.example" in line, \
                f"Dockerfile 不应该 COPY .env：{line}"

    def test_no_hardcoded_key_in_dockerfile(self):
        """Dockerfile 不能含 sk- 开头的真实 key。"""
        import re
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        # sk- 后跟至少 20 个字符才是真 key（避免误报）
        key_pattern = re.compile(r"sk-[a-zA-Z0-9]{20,}")
        assert not key_pattern.search(content), "Dockerfile 含硬编码 API key"

    def test_no_hardcoded_key_in_compose(self):
        """docker-compose.yml 不能含 sk- 开头的真实 key（只能 env_file 引用）。"""
        import re
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        key_pattern = re.compile(r"sk-[a-zA-Z0-9]{20,}")
        assert not key_pattern.search(content), "compose 文件含硬编码 API key"

    def test_no_hardcoded_key_in_entrypoint(self):
        """entrypoint 脚本不能含真实 key。"""
        import re
        for fname in ["docker-entrypoint.sh", "gunicorn.conf.py"]:
            content = (PROJECT_ROOT / fname).read_text()
            key_pattern = re.compile(r"sk-[a-zA-Z0-9]{20,}")
            assert not key_pattern.search(content), f"{fname} 含硬编码 API key"

    def test_env_in_gitignore(self):
        """`.env` 必须在 .gitignore 里（防提交到 git）。"""
        content = (PROJECT_ROOT / ".gitignore").read_text()
        assert ".env" in content


# ---------------------------------------------------------------------------
# Dockerfile 多阶段构建
# ---------------------------------------------------------------------------

class TestDockerfile:
    def test_uses_multi_stage(self):
        """应该使用多阶段构建（至少 2 个 FROM）。"""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        from_count = content.count("FROM ")
        assert from_count >= 2, f"多阶段构建应该至少 2 个 FROM，实际 {from_count}"

    def test_uses_slim_base(self):
        """应该用 slim 基础镜像，不用完整 python 镜像。"""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "python:3.13-slim" in content, "应该用 python:3.13-slim 基础镜像"

    def test_has_healthcheck(self):
        """Dockerfile 应该配置 HEALTHCHECK。"""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "HEALTHCHECK" in content

    def test_runs_as_non_root(self):
        """应该用非 root 用户运行（USER 指令）。"""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "USER " in content, "应该有 USER 指令切换非 root 用户"

    def test_has_tini_or_init(self):
        """应该用 tini 或 init 处理信号（容器僵尸进程问题）。"""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "tini" in content or "init" in content

    def test_python_optimizations(self):
        """应该设置 PYTHONUNBUFFERED 和 PYTHONDONTWRITEBYTECODE。"""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "PYTHONUNBUFFERED=1" in content
        assert "PYTHONDONTWRITEBYTECODE=1" in content


# ---------------------------------------------------------------------------
# docker-compose 配置
# ---------------------------------------------------------------------------

class TestComposeConfig:
    def test_compose_yaml_loads(self):
        """compose 文件应该是合法 YAML。"""
        import yaml
        with open(PROJECT_ROOT / "docker-compose.yml") as f:
            data = yaml.safe_load(f)
        assert "services" in data
        assert "app" in data["services"]

    def test_compose_has_env_file(self):
        """必须用 env_file 注入密钥（不是 hardcode）。"""
        import yaml
        with open(PROJECT_ROOT / "docker-compose.yml") as f:
            data = yaml.safe_load(f)
        app = data["services"]["app"]
        assert "env_file" in app, "应该用 env_file 注入密钥"
        assert ".env" in app["env_file"]

    def test_compose_has_healthcheck(self):
        """compose 应该配置 healthcheck。"""
        import yaml
        with open(PROJECT_ROOT / "docker-compose.yml") as f:
            data = yaml.safe_load(f)
        assert "healthcheck" in data["services"]["app"]

    def test_compose_has_volumes(self):
        """compose 应该挂载 data volume（持久化 ChromaDB）。"""
        import yaml
        with open(PROJECT_ROOT / "docker-compose.yml") as f:
            data = yaml.safe_load(f)
        volumes = data["services"]["app"].get("volumes", [])
        assert any("data" in str(v) for v in volumes), \
            "应该挂载 data 目录持久化索引"

    def test_compose_restart_policy(self):
        """应该有 restart 策略。"""
        import yaml
        with open(PROJECT_ROOT / "docker-compose.yml") as f:
            data = yaml.safe_load(f)
        assert data["services"]["app"].get("restart") == "unless-stopped"


# ---------------------------------------------------------------------------
# gunicorn 配置
# ---------------------------------------------------------------------------

class TestGunicornConfig:
    def test_gunicorn_config_loads(self):
        """gunicorn.conf.py 应该能被 Python 加载。"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gunicorn_conf", PROJECT_ROOT / "gunicorn.conf.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.workers >= 1
        assert mod.worker_class == "uvicorn.workers.UvicornWorker"
        assert mod.timeout >= 60  # LLM 调用要长 timeout

    def test_gunicorn_in_requirements(self):
        """gunicorn 应该在 requirements.txt 里。"""
        content = (PROJECT_ROOT / "requirements.txt").read_text()
        assert "gunicorn" in content


# ---------------------------------------------------------------------------
# .env.example 完整性
# ---------------------------------------------------------------------------

class TestEnvExample:
    """`.env.example` 应该包含所有必需配置项。"""

    @pytest.mark.parametrize("key", [
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "LLM_MODEL",
        "EMBEDDING_MODEL",
        "EMBEDDING_PROVIDER",
        "CHROMA_PERSIST_DIR",
        "CHROMA_COLLECTION",
    ])
    def test_env_example_has_key(self, key):
        content = (PROJECT_ROOT / ".env.example").read_text()
        assert key in content, f".env.example 缺少配置项 {key}"

    def test_env_example_uses_placeholder(self):
        """`.env.example` 应该用占位符，不能含真实 key。"""
        import re
        content = (PROJECT_ROOT / ".env.example").read_text()
        key_pattern = re.compile(r"sk-[a-zA-Z0-9]{20,}")
        assert not key_pattern.search(content), \
            ".env.example 含疑似真实 key"


# ---------------------------------------------------------------------------
# README 完整性
# ---------------------------------------------------------------------------

class TestReadme:
    def test_readme_has_installation(self):
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "docker compose up" in content.lower() or "docker-compose up" in content.lower()

    def test_readme_has_api_docs(self):
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "/api/v1/diet-plan" in content

    def test_readme_has_security_section(self):
        """README 应该有安全说明（API key 保护）。"""
        content = (PROJECT_ROOT / "README.md").read_text()
        # 中文或英文
        assert "安全" in content or "Security" in content or "security" in content

    def test_readme_mentions_learning_stages(self):
        """README 应该提到 6 阶段学习路径。"""
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "Stage" in content
