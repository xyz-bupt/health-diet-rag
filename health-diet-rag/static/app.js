// 健康饮食 RAG 前端逻辑

const form = document.getElementById('profile-form');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const progressEl = document.getElementById('progress');
const streamBtn = document.getElementById('generate-btn');
const syncBtn = document.getElementById('generate-sync-btn');

// 收集表单数据
function getProfile() {
    const formData = new FormData(form);
    const profile = {};
    for (const [key, value] of formData.entries()) {
        if (['height_cm', 'weight_kg', 'age'].includes(key)) {
            profile[key] = parseFloat(value);
        } else {
            profile[key] = value;
        }
    }
    return profile;
}

// 显示状态消息
function setStatus(message, type = 'info') {
    statusEl.textContent = message;
    statusEl.className = 'status ' + type;
}

// 清空状态
function clearStatus() {
    statusEl.textContent = '';
    statusEl.className = 'status';
}

// 重置进度指示器
function resetProgress() {
    progressEl.classList.remove('hidden');
    progressEl.querySelectorAll('.step').forEach(s => {
        s.classList.remove('active', 'done');
    });
}

// 标记某个步骤为进行中
function setActiveStep(nodeName) {
    const step = progressEl.querySelector(`[data-step="${nodeName}"]`);
    if (step) step.classList.add('active');
}

// 标记某个步骤为完成
function setDoneStep(nodeName) {
    const step = progressEl.querySelector(`[data-step="${nodeName}"]`);
    if (step) {
        step.classList.remove('active');
        step.classList.add('done');
    }
}

// 渲染单个 Node 的结果
function renderNodeResult(nodeName, data) {
    // 后端 SSE 推送格式：{"node": "health_node", "result": {"health": {...}}}
    // LangGraph 的 state 更新是 {field_name: value}，需要拆出内层对象
    const unwrapKey = {
        'health_node': 'health',
        'nutrition_node': 'nutrition',
        'recipe_node': 'recipe',
        'exercise_node': 'exercise',
        'supervisor_node': 'final_plan',
    };
    const key = unwrapKey[nodeName];
    if (key && data && typeof data === 'object' && data[key]) {
        data = data[key];
    }

    // 移除该节点的占位（如果有）
    let card = document.getElementById(`card-${nodeName}`);
    if (!card) {
        card = document.createElement('div');
        card.id = `card-${nodeName}`;
        card.className = 'result-card';
        resultEl.appendChild(card);
    }

    switch (nodeName) {
        case 'health_node':
            card.innerHTML = renderHealth(data);
            break;
        case 'nutrition_node':
            card.innerHTML = renderNutrition(data);
            break;
        case 'recipe_node':
            card.innerHTML = renderRecipe(data);
            break;
        case 'exercise_node':
            card.innerHTML = renderExercise(data);
            break;
        case 'supervisor_node':
            card.innerHTML = renderSummary(data);
            break;
    }
}

function llmBadge(used) {
    return used
        ? '<span class="llm-badge real">DeepSeek</span>'
        : '<span class="llm-badge mock">Mock</span>';
}

function renderHealth(h) {
    return `
        <h3>🩺 健康评估 ${llmBadge(h.llm_used)}</h3>
        <div class="key-metrics">
            <div class="metric"><span class="label">BMI</span><span class="value">${h.bmi} (${h.bmi_category})</span></div>
            <div class="metric"><span class="label">BMR</span><span class="value">${h.bmr.toFixed(0)} kcal</span></div>
            <div class="metric"><span class="label">TDEE</span><span class="value">${h.tdee.toFixed(0)} kcal</span></div>
            <div class="metric"><span class="label">目标热量</span><span class="value">${h.target_calories.toFixed(0)} kcal</span></div>
        </div>
        <p>${h.summary}</p>
    `;
}

function renderNutrition(n) {
    const mc = n.meal_calories;
    return `
        <h3>🥗 营养规划</h3>
        <div class="key-metrics">
            <div class="metric"><span class="label">每日蛋白</span><span class="value">${n.macros_daily.protein_g.toFixed(0)} g</span></div>
            <div class="metric"><span class="label">每日碳水</span><span class="value">${n.macros_daily.carbs_g.toFixed(0)} g</span></div>
            <div class="metric"><span class="label">每日脂肪</span><span class="value">${n.macros_daily.fat_g.toFixed(0)} g</span></div>
            <div class="metric"><span class="label">饮水</span><span class="value">${n.hydration_ml} ml</span></div>
        </div>
        <p><strong>三餐分配:</strong> 早 ${mc.breakfast.toFixed(0)} / 午 ${mc.lunch.toFixed(0)} / 晚 ${mc.dinner.toFixed(0)} / 加餐 ${mc.snack.toFixed(0)} kcal</p>
        ${n.timing_tips && n.timing_tips.length ? `
            <p><strong>进食时机:</strong></p>
            <ul>${n.timing_tips.map(t => `<li>${t}</li>`).join('')}</ul>
        ` : ''}
    `;
}

function renderRecipe(r) {
    const meals = r.meals.map(m => `
        <div class="meal">
            <div class="meal-name">${mealTypeLabel(m.meal_type)}: ${m.name} (${m.calories.toFixed(0)} kcal)</div>
            <div class="ingredients">${m.ingredients.join('、')}</div>
        </div>
    `).join('');
    return `
        <h3>🍽️ 三餐菜谱 ${llmBadge(r.llm_used)}</h3>
        <p><strong>总热量:</strong> ${r.total_calories.toFixed(0)} kcal</p>
        ${meals}
        ${r.variety_note ? `<p><em>${r.variety_note}</em></p>` : ''}
    `;
}

function mealTypeLabel(t) {
    return {breakfast: '早餐', lunch: '午餐', dinner: '晚餐', snack: '加餐'}[t] || t;
}

function renderExercise(e) {
    const sessions = e.weekly_sessions.map(s => `
        <li>${s.day}: ${s.description} (${s.type}, ${s.duration_min}min, ${s.intensity})</li>
    `).join('');
    return `
        <h3>🏃 运动建议</h3>
        <div class="key-metrics">
            <div class="metric"><span class="label">每周消耗</span><span class="value">${e.weekly_calories_burned.toFixed(0)} kcal</span></div>
        </div>
        <ul>${sessions}</ul>
        ${e.tips && e.tips.length ? `
            <p><strong>注意事项:</strong></p>
            <ul>${e.tips.map(t => `<li>${t}</li>`).join('')}</ul>
        ` : ''}
    `;
}

function renderSummary(d) {
    return `
        <h3>📋 完整方案摘要 ${llmBadge(d.llm_used)}</h3>
        <p>${d.summary}</p>
        <p><strong>🔑 最优先行动:</strong></p>
        <ul>${d.key_actions.map(a => `<li>${a}</li>`).join('')}</ul>
    `;
}

// 显示完整方案（同步接口）
function renderFullPlan(plan) {
    resultEl.innerHTML = '';
    renderNodeResult('health_node', plan.health);
    renderNodeResult('nutrition_node', plan.nutrition);
    renderNodeResult('recipe_node', plan.recipe);
    renderNodeResult('exercise_node', plan.exercise);
    renderNodeResult('supervisor_node', plan);
}

// ----- 流式生成 -----

async function generateStream() {
    const profile = getProfile();
    resultEl.innerHTML = '<p class="placeholder">流式接收中...</p>';
    resetProgress();
    setStatus('🚀 启动工作流（流式）...', 'info');
    streamBtn.disabled = true;
    syncBtn.disabled = true;

    try {
        const response = await fetch('/api/v1/diet-plan/stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(profile),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error?.message || `HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let currentNode = null;

        // 清空结果区，准备逐个追加
        resultEl.innerHTML = '';

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, {stream: true});
            const lines = buffer.split('\n');
            buffer = lines.pop(); // 最后不完整的行留作下次

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const payload = line.slice(6).trim();
                if (payload === '[DONE]') {
                    continue;
                }
                try {
                    const event = JSON.parse(payload);
                    if (event.error) {
                        setStatus(`❌ 错误: ${event.error.message}`, 'error');
                        continue;
                    }
                    // 标记前一个节点完成
                    if (currentNode) setDoneStep(currentNode);
                    currentNode = event.node;
                    setActiveStep(currentNode);
                    // 渲染当前节点的结果
                    renderNodeResult(currentNode, event.result);
                } catch (e) {
                    console.warn('解析 SSE 失败:', e, payload);
                }
            }
        }
        if (currentNode) setDoneStep(currentNode);
        setStatus('✅ 方案生成完成', 'success');
    } catch (err) {
        setStatus(`❌ ${err.message}`, 'error');
    } finally {
        streamBtn.disabled = false;
        syncBtn.disabled = false;
    }
}

// ----- 同步生成 -----

async function generateSync() {
    const profile = getProfile();
    resultEl.innerHTML = '<p class="placeholder">⏳ 工作流执行中（约 1-3 秒）...</p>';
    progressEl.classList.add('hidden');
    setStatus('⏳ 同步生成中...', 'info');
    streamBtn.disabled = true;
    syncBtn.disabled = true;

    try {
        const response = await fetch('/api/v1/diet-plan', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(profile),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error?.message || `HTTP ${response.status}`);
        }

        const plan = await response.json();
        renderFullPlan(plan);
        const cached = 'cache 命中' ? '' : '';
        setStatus(`✅ 完成（${plan.llm_used ? 'DeepSeek 真实 LLM' : 'Mock 兜底'}）`, 'success');
    } catch (err) {
        setStatus(`❌ ${err.message}`, 'error');
    } finally {
        streamBtn.disabled = false;
        syncBtn.disabled = false;
    }
}

// 绑定事件
streamBtn.addEventListener('click', generateStream);
syncBtn.addEventListener('click', generateSync);
