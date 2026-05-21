"""
Multi-Agent Decision Support System (MAS-DSS)
"""

import json
import os
import re
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

_api_key = os.environ.get("OPENAI_API_KEY", "")
if not _api_key:
    print("WARNING: OPENAI_API_KEY not set.")

client = OpenAI(api_key=_api_key) if _api_key else None
MODEL = "gpt-4o"
app = FastAPI(title="MAS Decision Support System", version="1.0.0")

DOMAIN_CONTEXTS = {
    "healthcare": {
        "label": "Healthcare",
        "context": "Focus on patient outcomes, clinical workflows, regulatory compliance (HIPAA, FDA), evidence-based medicine, health equity, and system sustainability.",
        "solution_types": "clinical protocols, digital health tools, care pathways, policy interventions, staff training programs, community health initiatives",
        "kpi_examples": "patient readmission rates, HCAHPS scores, time-to-treatment, cost per episode of care, staff burnout indices"
    },
    "food_systems": {
        "label": "Food Systems",
        "context": "Focus on supply chain resilience, food security, nutrition equity, sustainability, smallholder livelihoods, and reducing food waste.",
        "solution_types": "supply chain interventions, community food programs, agricultural policies, distribution models, technology platforms, certification schemes",
        "kpi_examples": "food insecurity rates, supply chain waste %, smallholder income, dietary diversity scores, carbon footprint per calorie"
    },
    "urban_planning": {
        "label": "Urban Planning",
        "context": "Focus on livability, mobility, housing equity, climate resilience, economic vitality, and inclusive community development.",
        "solution_types": "zoning policies, infrastructure projects, community programs, mobility services, green space initiatives, affordable housing schemes",
        "kpi_examples": "commute times, affordable housing units, green space per capita, air quality index, community satisfaction scores"
    },
    "business_strategy": {
        "label": "Business Strategy",
        "context": "Focus on competitive differentiation, operational efficiency, market positioning, talent retention, innovation capacity, and financial sustainability.",
        "solution_types": "products, services, processes, organizational programs, market policies, business model innovations",
        "kpi_examples": "revenue growth, market share, NPS, employee retention, operational cost reduction, time-to-market"
    }
}

class ProblemRequest(BaseModel):
    problem_statement: str
    domain: str = "healthcare"
    organization_context: Optional[str] = ""
    constraints: Optional[str] = ""

def get_interpreter_prompt(domain):
    ctx = DOMAIN_CONTEXTS.get(domain, DOMAIN_CONTEXTS["healthcare"])
    return f"""You are the Interpreter Agent in a Multi-Agent Decision Support System.
Your role: Analyze the problem statement and structure it into a precise problem packet.
Domain context: {ctx['context']}
Return ONLY valid JSON:
{{
  "undesired_state": {{"summary": "one sentence","observable_conditions": ["..."],"measurable_indicators": ["..."],"affected_entities": ["..."]}},
  "causal_factors": {{"present_causes": ["..."],"absent_causes": ["..."]}},
  "desired_state": {{"target_condition": "...","success_criteria": ["..."],"constraints": ["..."]}},
  "classification": {{"problem_type": "...","urgency": "...","complexity": "...","primary_stakeholders": ["..."]}},
  "refined_problem_statement": "A precise 2-3 sentence restatement"
}}"""

def get_retriever_prompt(domain):
    ctx = DOMAIN_CONTEXTS.get(domain, DOMAIN_CONTEXTS["healthcare"])
    return f"""You are the Retriever Agent in a Multi-Agent Decision Support System.
Your role: Retrieve and synthesize relevant knowledge, frameworks, evidence, and precedents.
Domain context: {ctx['context']}
Return ONLY valid JSON:
{{
  "knowledge_base": {{
    "evidence_summary": "2-3 paragraph synthesis",
    "proven_interventions": [{{"intervention": "...","context": "...","evidence_strength": "strong/moderate/weak","source_type": "..."}}],
    "analogous_cases": [{{"domain": "...","case": "...","transferable_insight": "..."}}],
    "applicable_frameworks": ["..."],
    "pitfalls_to_avoid": ["..."]
  }},
  "retrieval_confidence": "high/medium/low",
  "knowledge_gaps": ["..."]
}}"""

def get_ideation_prompt(domain):
    ctx = DOMAIN_CONTEXTS.get(domain, DOMAIN_CONTEXTS["healthcare"])
    return f"""You are the Ideation Agent in a Multi-Agent Decision Support System.
Your role: Generate 6-8 creative, diverse, actionable solution ideas.
Domain context: {ctx['context']}
Solution types: {ctx['solution_types']}
Return ONLY valid JSON:
{{
  "ideas": [
    {{
      "id": "IDEA-001",
      "title": "...",
      "solution_type": "product/service/process/program/policy/paradigm",
      "description": "2-3 sentences",
      "target_cause": "...",
      "mechanism": "...",
      "analogy_source": "...",
      "implementation_sketch": "3-5 steps",
      "novelty_rationale": "...",
      "quick_feasibility": "high/medium/low"
    }}
  ],
  "ideation_methods_used": ["..."],
  "recommended_bundle": "..."
}}"""

def get_evaluator_prompt(domain):
    ctx = DOMAIN_CONTEXTS.get(domain, DOMAIN_CONTEXTS["healthcare"])
    return f"""You are the Evaluator Agent in a Multi-Agent Decision Support System.
Your role: Evaluate each idea across 11 dimensions and produce a ranked scorecard.
Domain context: {ctx['context']}
KPIs: {ctx['kpi_examples']}
Score each 0-100. Weights: efficacy:0.20, feasibility:0.15, financial:0.12, desirability:0.10, equity:0.08, risk:0.10, scalability:0.07, time_to_impact:0.05, creativity_novelty:0.05, creativity_non_obviousness:0.04, creativity_value:0.04
IMPORTANT: Return ONLY valid JSON with no trailing commas.
{{
  "evaluations": [
    {{
      "idea_id": "IDEA-001",
      "idea_title": "...",
      "scores": {{
        "efficacy": {{"score": 75, "rationale": "..."}},
        "feasibility": {{"score": 75, "rationale": "..."}},
        "financial": {{"score": 75, "rationale": "..."}},
        "desirability": {{"score": 75, "rationale": "..."}},
        "equity": {{"score": 75, "rationale": "..."}},
        "risk": {{"score": 75, "rationale": "..."}},
        "scalability": {{"score": 75, "rationale": "..."}},
        "time_to_impact": {{"score": 75, "rationale": "..."}},
        "creativity_novelty": {{"score": 75, "rationale": "..."}},
        "creativity_non_obviousness": {{"score": 75, "rationale": "..."}},
        "creativity_value": {{"score": 75, "rationale": "..."}}
      }},
      "weighted_score": 75,
      "key_strengths": ["..."],
      "key_risks": ["..."],
      "recommendation": "Scale/Pilot/Pivot/Kill"
    }}
  ],
  "ranking": ["IDEA-001"],
  "top_recommendation": "IDEA-001",
  "recommended_bundle": {{"ideas": ["IDEA-001"], "bundle_rationale": "...", "bundle_score": 75}}
}}"""

def get_synthesizer_prompt(domain, org_context):
    ctx = DOMAIN_CONTEXTS.get(domain, DOMAIN_CONTEXTS["healthcare"])
    return f"""You are the Synthesizer Agent in a Multi-Agent Decision Support System.
Your role: Integrate all agent outputs into an executive-quality Decision Support Report.
Domain context: {ctx['context']}
Organization context: {org_context or 'Not specified'}
Return ONLY valid JSON:
{{
  "executive_summary": "...",
  "problem_synthesis": "...",
  "top_recommendations": [
    {{
      "rank": 1,
      "idea_id": "IDEA-001",
      "title": "...",
      "why_selected": "...",
      "implementation_roadmap": {{"immediate": ["..."],"thirty_days": ["..."],"ninety_days": ["..."]}},
      "expected_outcomes": ["..."],
      "risks_and_mitigations": [{{"risk": "...","mitigation": "..."}}]
    }}
  ],
  "recommended_bundle": {{"description": "...","combined_theory_of_change": "..."}},
  "implementation_timeline": {{"weeks_1_2": ["..."],"months_1_3": ["..."],"months_3_12": ["..."]}},
  "success_metrics": [{{"metric": "...","target": "...","measurement_cadence": "...","decision_trigger": "..."}}],
  "human_decisions_required": ["..."],
  "next_steps": [{{"action": "...","owner_role": "...","timeline": "...","priority": "high/medium/low"}}]
}}"""

def repair_json(content):
    """Attempt to repair common JSON issues."""
    content = content.strip()
    content = content[content.find("{"):content.rfind("}")+1]
    content = re.sub(r',\s*}', '}', content)
    content = re.sub(r',\s*]', ']', content)
    return content

def run_agent(system_prompt, user_message, agent_name):
    if not client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server.")
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
            max_tokens=3000
        )
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens
        content = repair_json(content)
        return json.loads(content), tokens
    except json.JSONDecodeError:
        # Second attempt with more aggressive repair
        try:
            content = repair_json(content)
            content = re.sub(r'[\x00-\x1f\x7f]', '', content)
            return json.loads(content), tokens
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{agent_name} returned invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{agent_name} failed: {str(e)}")

@app.post("/api/analyze")
async def analyze_problem(request: ProblemRequest):
    domain = request.domain if request.domain in DOMAIN_CONTEXTS else "healthcare"
    results = {"domain": domain, "total_tokens": 0, "status": "running"}

    interp_output, t = run_agent(get_interpreter_prompt(domain),
        f"Problem: {request.problem_statement}\nContext: {request.organization_context}\nConstraints: {request.constraints}\nDomain: {DOMAIN_CONTEXTS[domain]['label']}",
        "Interpreter Agent")
    results["interpreter"] = interp_output
    results["total_tokens"] += t

    retriev_output, t = run_agent(get_retriever_prompt(domain),
        f"Problem Packet:\n{json.dumps(interp_output, indent=2)}\nOriginal: {request.problem_statement}",
        "Retriever Agent")
    results["retriever"] = retriev_output
    results["total_tokens"] += t

    idea_output, t = run_agent(get_ideation_prompt(domain),
        f"Problem Packet:\n{json.dumps(interp_output, indent=2)}\nKnowledge:\n{json.dumps(retriev_output, indent=2)}",
        "Ideation Agent")
    results["ideation"] = idea_output
    results["total_tokens"] += t

    eval_output, t = run_agent(get_evaluator_prompt(domain),
        f"Desired State:\n{json.dumps(interp_output.get('desired_state', {}), indent=2)}\nIdeas:\n{json.dumps(idea_output, indent=2)}",
        "Evaluator Agent")
    results["evaluator"] = eval_output
    results["total_tokens"] += t

    synth_output, t = run_agent(get_synthesizer_prompt(domain, request.organization_context or ""),
        f"INTERPRETER:\n{json.dumps(interp_output, indent=2)}\nRETRIEVER:\n{json.dumps(retriev_output, indent=2)}\nIDEATION:\n{json.dumps(idea_output, indent=2)}\nEVALUATOR:\n{json.dumps(eval_output, indent=2)}",
        "Synthesizer Agent")
    results["synthesizer"] = synth_output
    results["total_tokens"] += t

    results["status"] = "complete"
    results["estimated_cost_usd"] = round(results["total_tokens"] * 0.000005, 4)
    return results

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": MODEL}

@app.get("/api/domains")
async def get_domains():
    return {k: v["label"] for k, v in DOMAIN_CONTEXTS.items()}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🧠 MAS Decision Support System")
    print(f"================================")
    print(f"Starting server at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
