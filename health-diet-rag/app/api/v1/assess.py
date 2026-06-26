"""
健康评估路由。

提供：
- POST /api/v1/assess：提交健康画像，返回评估结果
"""

from fastapi import APIRouter

from app.agents.health import HealthAssessmentAgent
from app.models.health import HealthAssessment, HealthProfile

router = APIRouter()


@router.post("/assess", response_model=HealthAssessment, summary="健康评估")
async def assess(profile: HealthProfile) -> HealthAssessment:
    """提交健康画像，返回个性化健康评估。

    - 输入：身高/体重/年龄/性别/活动量/目标
    - 输出：BMI / BMR / TDEE / 目标热量 / 宏量营养素 / 摘要 / 建议

    本接口的核心特性：
    1. **数学计算完全确定性**：BMI/BMR/TDEE 由代码算，结果 100% 可复现
    2. **LLM 解读可降级**：未配 DEEPSEEK_API_KEY 时返回规则版摘要（llm_used=false）
    """
    agent = HealthAssessmentAgent()
    return agent.assess(profile)
